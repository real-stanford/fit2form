import torch
from torch import save, cat, stack, no_grad, set_num_threads, is_tensor, tensor
from torch.nn import BCELoss, MSELoss, L1Loss
from torch.cuda import (
    device_count,
    set_device,
    is_available as is_cuda_available
)
from torch.utils.data import DataLoader, Subset
from torch.distributed import init_process_group, reduce as dist_reduce
from os.path import exists
from os import mkdir
from time import time

from utils import (
    dicts_get,
    seed_all,
    Logger,
    setup_pretrain
)
import numpy as np
from tqdm import tqdm
from typing import List
from environment import GraspSimulationEnv
from learning import (
    ObjectDataset,
    GraspDataset,
    GripperDesigner,
    get_combined_loader,
    get_balanced_cotrain_loader,
    get_loader,
    get_design_objective_indices,
    GraspDatasetType,
)
from distribute import Pool
from multiprocessing import cpu_count
from pprint import pprint


def extend_grasp_dataset(
        epoch: int,
        object_dataset: ObjectDataset,
        grasp_dataset: GraspDataset,
        net: GripperDesigner,
        env_pool: Pool,
        logger: Logger,
        grippers_directory: str,
        new_fingers_count: int):
    start_extend_grasp_dataset = time()
    new_grasp_stats = {
        'base_connected': [],
        'failed_gripper_creation': [],
        'single_connected_component': [],
        'grasp_scores': [],
    }
    newly_added_indices = list()  # list of indices for newly added datapoints in cotrain dataset
    while len(new_grasp_stats['grasp_scores']) < new_fingers_count:
        grasp_object = object_dataset.sample()
        print(f'\t1. Sampled {len(grasp_object)} grasp objects')
        grasp_object_tsdfs = stack(
            dicts_get(grasp_object, 'tsdf')
        ).to(net.device, non_blocking=True)
        with no_grad():
            left_finger_gpu, right_finger_gpu = \
                net.create_fingers(grasp_object_tsdfs)
        print('\t2. Generated fingers')
        left_finger = left_finger_gpu.cpu().clone()
        right_finger = right_finger_gpu.cpu().clone()
        # 2. Evaluate new fingers
        start_grasp_eval_time = time()
        gripper_output_paths = ['{}/{:06}'.format(
            grippers_directory, len(grasp_dataset) + i)
            for i in range(len(grasp_object))]
        grasp_results = env_pool.map(
            exec_fn=lambda env, args:
            env.compute_finger_grasp_score.remote(*args),
            iterable=zip(
                left_finger,
                right_finger,
                dicts_get(grasp_object, 'urdf_path'),
                gripper_output_paths))
        print('\t3. Evaluated fingers ({:.01f}s)'.format(
            float(time() - start_grasp_eval_time)))
        # Collect stats
        grasp_scores = dicts_get(grasp_results, 'score')
        new_grasp_stats['grasp_scores'].extend(
            list(filter(None, grasp_scores)))
        new_grasp_stats['base_connected'].extend(
            dicts_get(grasp_results, 'base_connected'))
        new_grasp_stats['failed_gripper_creation'].extend(
            list(filter(None,
                        dicts_get(
                            grasp_results, 'created_grippers_failed'))))
        new_grasp_stats['single_connected_component'].extend(
            list(filter(None,
                        dicts_get(
                            grasp_results, 'single_connected_component'))))

        # 4. Add new grasp results to grasp result training dataset
        tmp_new_indices = grasp_dataset.extend([
        {
            'grasp_object_tsdf_path': grasp_object_path,
                'left_finger_tsdf': left_finger,
                'right_finger_tsdf': right_finger,
                'grasp_result': np.array(list(grasp_score.values()))
            }
            for grasp_object_path, left_finger, right_finger, grasp_score
            in zip(dicts_get(grasp_object, 'tsdf_path'),
                left_finger,
                right_finger,
                grasp_scores)
            if grasp_score is not None
        ])
        newly_added_indices.extend(tmp_new_indices)

    log_data = {
        'Grasp_Scores/Success':
        np.mean(dicts_get(new_grasp_stats['grasp_scores'], 'success')),
        'Grasp_Scores/Stability_1':
        np.mean(dicts_get(new_grasp_stats['grasp_scores'], 'stability_1')),
        'Grasp_Scores/Stability_2':
        np.mean(dicts_get(new_grasp_stats['grasp_scores'], 'stability_2')),
        'Grasp_Scores/Stability_3':
        np.mean(dicts_get(new_grasp_stats['grasp_scores'], 'stability_3')),
        'Grasp_Scores/Stability_4':
        np.mean(dicts_get(new_grasp_stats['grasp_scores'], 'stability_4')),
        'Finger_Stats/Single_Connected_Component':
        1.0 if len(new_grasp_stats['single_connected_component']) == 0
        else np.mean(new_grasp_stats['single_connected_component']),
        'Finger_Stats/Base_Connected':
        1.0 if len(new_grasp_stats['base_connected']) == 0
        else np.mean(new_grasp_stats['base_connected']),
        'Finger_Stats/Mesh_Dump_Failures':
        0.0 if len(new_grasp_stats['failed_gripper_creation']) == 0
        else np.mean(new_grasp_stats['failed_gripper_creation']),
    }
    pprint(log_data)
    logger.log(
        data=log_data,
        step=epoch
    )
    print(f'Time taken for extend grasp dataset: {float(time()-start_extend_grasp_dataset):.1f}s')
    return newly_added_indices


def compute_fitness_loss(net,
                         grasp_object,
                         left_finger,
                         right_finger,
                         grasp_result,
                         fitness_criterion,
                         return_fitness_pred=False):
    grasp_result = grasp_result.to(net.device, non_blocking=True)
    fitness_pred = net.evaluate_fingers(
        grasp_object,
        left_finger,
        right_finger
    )
    if return_fitness_pred:
        return fitness_criterion(
            fitness_pred.float(),
            grasp_result.float()), fitness_pred
    return fitness_criterion(
        fitness_pred.float(),
        grasp_result.float())


def compute_generator_loss(net,
                           grasp_object,
                           return_fitness_pred=True):
    new_left_finger, new_right_finger = net.create_fingers(
        grasp_object)
    training = net.training
    net.eval()
    new_fitness_prediction = net.evaluate_fingers(
        grasp_object,
        new_left_finger,
        new_right_finger)
    net.train(mode=training)
    if return_fitness_pred:
        return -new_fitness_prediction.mean(), new_fitness_prediction
    return -new_fitness_prediction.mean()


def compute_generator_loss_imprint_gt(net, generator_criterion, batch):
    grasp_object, gt_left_finger, gt_right_finger, _, _ = batch
    gt_left_finger = gt_left_finger.to(net.device, non_blocking=True)
    gt_right_finger = gt_right_finger.to(net.device, non_blocking=True)
    new_left_finger, new_right_finger = net.create_fingers(
        grasp_object)
    loss_left = generator_criterion(new_left_finger.float(), gt_left_finger.float())
    loss_right = generator_criterion(new_right_finger.float(), gt_right_finger.float())
    return (loss_left + loss_right) / 2


def optimize_generator(
    data_loader: DataLoader,
    net: GripperDesigner,
    generator_criterion,
    logger: Logger,
    tqdm_prefix='',
    optimize=True):
    generator_losses = []
    with tqdm(data_loader, dynamic_ncols=True) as batch_pbar:
        for batch in batch_pbar:
            generator_loss = compute_generator_loss_imprint_gt(
                net, generator_criterion, batch)
            if optimize:
                net.gn_optimizer.zero_grad()
                generator_loss.backward()
                net.gn_optimizer.step()
                net.stats['gn_update_steps'] += 1
            generator_losses.append(generator_loss)
            batch_pbar.set_description(
                f'{tqdm_prefix} G: {generator_loss.item():.04f}')
    return stack(generator_losses).detach().mean()


def optimize_network(data_loader: DataLoader,
                     net: GripperDesigner,
                     fitness_criterion,
                     logger: Logger,
                     tqdm_prefix='',
                     get_fitness_loss=True,
                     optimize_fitness_net=True,
                     get_generator_loss=True,
                     optimize_generator_net=True,
                     fitness_predictions=None,
                     max_optimization_steps=None):
    assert get_fitness_loss or get_generator_loss
    reduced_fitness_losses = []
    fitness_losses = [] # list of (batch of fn grasp success loss)s
    fitness_gts = []  # list of (batch of gt grasp success)s
    fitness_preds = []  # list of (batch of fitness prediction for grasp success)s
    dataset_types = []  # list of (batch of datapoint is cotrain / pretrain)s
    generator_losses = []
    with tqdm(data_loader,
              dynamic_ncols=True,
              smoothing=0.05,
              total=max_optimization_steps) as batch_pbar:
        for i, batch in enumerate(batch_pbar):
            tqdm_desc = f'{tqdm_prefix} '
            if is_tensor(batch):
                grasp_object = batch
                get_fitness_loss = False
            else:
                grasp_object, left_finger, right_finger, grasp_result, grasp_dataset_type = batch
            if get_fitness_loss:
                fitness_loss, fitness_pred = compute_fitness_loss(
                    net, grasp_object, left_finger,
                    right_finger, grasp_result, fitness_criterion,
                    return_fitness_pred=True)

                fitness_losses.append(fitness_loss.detach().cpu().numpy())
                fitness_gts.append(grasp_result[:, 0].detach().cpu().numpy())
                fitness_preds.append(fitness_pred[:, 0].detach().cpu().numpy())
                dataset_types.append(grasp_dataset_type.detach().cpu().numpy())

                reduced_fitness_loss = fitness_loss.mean()
                reduced_fitness_losses.append(reduced_fitness_loss.detach().cpu())
                tqdm_desc += 'F: {:.04f} '.format(reduced_fitness_loss.item())
                if optimize_fitness_net:
                    net.fn_optimizer.zero_grad()
                    reduced_fitness_loss.backward()
                    net.fn_optimizer.step()
                    net.stats['fn_update_steps'] += 1
                    if logger is not None:
                        logger.log(
                            data={
                                'Training/Fitness_Loss':
                                reduced_fitness_loss.item()
                            },
                            step=net.stats['fn_update_steps'])
            if get_generator_loss:
                generator_loss, new_fitness_prediction = \
                    compute_generator_loss(net, grasp_object)
                generator_losses.append(generator_loss)
                if fitness_predictions:
                    fitness_predictions.append(new_fitness_prediction)
                tqdm_desc += 'G: {:.04f} '.format(generator_loss.item())
                if optimize_generator_net:
                    net.gn_optimizer.zero_grad()
                    generator_loss.backward()
                    net.gn_optimizer.step()
                    net.stats['gn_update_steps'] += 1
                    if logger is not None:
                        logger.log(
                            data={
                                'Training/Generator_Loss':
                                generator_loss.item(),
                                'Fitness_Predictions/Success':
                                new_fitness_prediction[:, 0].mean()
                            },
                            step=net.stats['gn_update_steps'])
            batch_pbar.set_description(tqdm_desc)
            if max_optimization_steps and i >= max_optimization_steps - 1:
                break
    log_text = 'Average '
    if len(reduced_fitness_losses) > 0:
        mean_fitness_loss = stack(reduced_fitness_losses).mean().item()
        log_text += 'F: {:.04f} '.format(mean_fitness_loss)
    else:
        mean_fitness_loss = None
    if len(generator_losses) > 0:
        mean_generator_loss = stack(generator_losses).detach().mean()
        log_text += 'G: {:.04f} '.format(mean_generator_loss.item())
    else:
        mean_generator_loss = None
    # print(log_text)
    return mean_fitness_loss, mean_generator_loss, fitness_gts, fitness_preds, dataset_types, fitness_losses


def cotrain(envs: List[GraspSimulationEnv],
            net: GripperDesigner,
            hyperparams: dict,
            object_dataset: ObjectDataset,
            cotrain_grasp_dataset: GraspDataset,
            pretrain_training_dataset: GraspDataset,
            logger: Logger,
            generate_new_grasps=True):
    if net.device == 'cpu':
        print("Running on CPU. Using 48 threads")
        set_num_threads(cpu_count())
    # reset stats
    for key in net.stats.keys():
        net.stats[key] = 0
    env_pool = Pool(envs)
    fitness_criterion = BCELoss(reduction='none')

    batch_size = hyperparams['cotrain_batch_size']

    grippers_directory = logger.logdir + '/grippers'

    def get_mixed_cotrain_pretrain_loader():
        if hyperparams['balanced_sampling']:
            return get_balanced_cotrain_loader(
                cotrain_dataset=cotrain_grasp_dataset,
                pretrain_dataset=pretrain_training_dataset,
                batch_size=batch_size)
        else:
            return get_combined_loader(
                get_loader(
                    dataset=cotrain_grasp_dataset,
                    batch_size=int(batch_size / 2)),
                get_loader(
                    dataset=pretrain_training_dataset,
                    batch_size=int(batch_size / 2)))

    get_fn_loader = get_mixed_cotrain_pretrain_loader

    get_gn_loader = get_mixed_cotrain_pretrain_loader

    # Pretrain dataset statistics
    pretrain_success_indices, pretrain_failure_indices = \
        pretrain_training_dataset.get_indices_distribution()
    logger.log_text({
        "pretrain_data/success_indices": len(pretrain_success_indices),
        "pretrain_data/failure_indices": len(pretrain_failure_indices)
    }, -1)

    if hyperparams['cotrain_optimize_gn_grasp_object']:
        def get_gn_loader():
            return get_loader(
                object_dataset,
                batch_size=batch_size,
                collate_fn=lambda batch:
                stack([item['tsdf'] for item in batch])
            )

    if not exists(grippers_directory):
        mkdir(grippers_directory)
    if 'fn_update_steps' not in net.stats:
        net.stats['fn_update_steps'] = 0
    if 'gn_update_steps' not in net.stats:
        net.stats['gn_update_steps'] = 0

    def get_fn_performance(data_loader, dataset_name, before_or_after):
        # Utility function for getting fn performance for logging purposes
        return optimize_network(
            data_loader=data_loader,
            net=net,
            fitness_criterion=L1Loss(reduction='none'),
            logger=None,
            tqdm_prefix=f'FN stats on {dataset_name} {before_or_after} updating FN weights|',
            get_fitness_loss=True,
            optimize_fitness_net=False,
            get_generator_loss=False,
            optimize_generator_net=False)

    net_train_started = False  # Helper flag for logging train start epoch
    for epoch in range(net.stats['epochs'] + 1, hyperparams['cotrain_epochs']):
        print('=' * 25 + ' EPOCH {} '.format(epoch) + '=' * 25)
        start_epoch_time = time()
        newly_added_indices = list()
        # ============================================
        # Geenrate new data
        # ============================================
        if generate_new_grasps:
            net.eval()
            newly_added_indices = extend_grasp_dataset(
                epoch, object_dataset, cotrain_grasp_dataset,
                net, env_pool, logger, grippers_directory,
                hyperparams['cotrain_new_fingers_per_epoch_count'])

        # ============================================
        # Collect cotrain dataset statistics
        # ============================================
        cotrain_success_indices = np.zeros(len(cotrain_grasp_dataset), dtype=np.bool)
        cotrain_failure_indices = np.zeros(len(cotrain_grasp_dataset), dtype=np.bool)
        tmp_si, tmp_fi = cotrain_grasp_dataset.get_indices_distribution()
        for i in tmp_si: 
            cotrain_success_indices[i] = True
        for i in tmp_fi: 
            cotrain_failure_indices[i] = True
        cotrain_old_indices = np.ones(len(cotrain_grasp_dataset), dtype=np.bool)
        cotrain_new_indices = np.zeros(len(cotrain_grasp_dataset), dtype=np.bool)
        for i in newly_added_indices:
            cotrain_old_indices[i] = False
            cotrain_new_indices[i] = True
        logger.log_scalars({"Cotrain_data_success_rate": {
            "old": np.sum(cotrain_old_indices & cotrain_success_indices) / np.sum(cotrain_old_indices)
            if np.sum(cotrain_old_indices) != 0 else 0,
            "new": np.sum(cotrain_new_indices & cotrain_success_indices) / np.sum(cotrain_new_indices)
            if np.sum(cotrain_new_indices) != 0 else 0,
            "full": np.mean(cotrain_success_indices)
            if len(cotrain_success_indices) != 0 else 0,
        }}, epoch)

        # ============================================
        # Add statistics before training Fitness network
        # ============================================
        def collect_all_fn_performance(before_or_after):
            # - fn performance on pretrain data
            since = time()
            dataset_name = "Pretrain"
            len_dataset = len(pretrain_training_dataset)
            random_subset_indices = np.random.choice(
                len_dataset,
                size=(min(len_dataset, 6000)),
                replace=False
            )
            fn_pretrain_loader = get_loader(
                dataset=Subset(pretrain_training_dataset, random_subset_indices),
                batch_size=batch_size,
                drop_last=False,
            )

            _, _, _, fn_pretrain_preds, _, fn_pretrain_losses = \
                get_fn_performance(fn_pretrain_loader, dataset_name, before_or_after)
            fn_pretrain_preds = np.hstack(fn_pretrain_preds)
            fn_pretrain_losses = np.vstack(fn_pretrain_losses)
            print(f"Time taken for collecting fn performance on {dataset_name} {before_or_after} updating fn weights: {time() - since} s")

            # - fn performance on cotrain data
            since = time()
            dataset_name = "Cotrain_full"
            _, _, _, fn_cotrain_full_preds, _, fn_cotrain_full_losses = get_fn_performance(
                get_loader(cotrain_grasp_dataset, batch_size, shuffle=False, drop_last=False), dataset_name, before_or_after)
            fn_cotrain_full_preds = np.hstack(fn_cotrain_full_preds)
            fn_cotrain_full_losses = np.vstack(fn_cotrain_full_losses)
            print(f"Time taken for collecting fitness loss on cotrain_dataset {before_or_after} updating fn weights: {time() - since} s")
            return {
                f"Pretrain_{before_or_after}": fn_pretrain_losses.mean()
                    if len(fn_pretrain_losses) != 0 else 0,
                f"cotrain_full_{before_or_after}": fn_cotrain_full_losses.mean()
                    if len(fn_cotrain_full_losses) != 0 else 0,
                f"cotrain_old_{before_or_after}": fn_cotrain_full_losses[cotrain_old_indices].mean()
                    if len(fn_cotrain_full_losses[cotrain_old_indices]) != 0 else 0,
                f"cotrain_new_{before_or_after}": fn_cotrain_full_losses[cotrain_new_indices].mean()
                    if len(fn_cotrain_full_losses[cotrain_new_indices]) != 0 else 0,
            }, {
                f"FN_Preds/Pretrain_{before_or_after}": fn_pretrain_preds
                    if len(fn_pretrain_preds) != 0 else 0,
                f"FN_Preds/cotrain_full_{before_or_after}": fn_cotrain_full_preds
                    if len(fn_cotrain_full_preds) != 0 else 0,
                f"FN_Preds/cotrain_old_{before_or_after}": fn_cotrain_full_preds[cotrain_old_indices]
                    if len(fn_cotrain_full_preds[cotrain_old_indices]) != 0 else 0,
                f"FN_Preds/cotrain_new_{before_or_after}": fn_cotrain_full_preds[cotrain_new_indices]
                    if len(fn_cotrain_full_preds[cotrain_new_indices]) != 0 else 0,
            }

        before_fn_losses_dict, before_fn_preds_dict =  collect_all_fn_performance("before")
        # ============================================
        # Train FN
        # ============================================
        if len(cotrain_grasp_dataset) < \
                hyperparams['cotrain_warmup_datapoints']:
            continue
        if not net_train_started:
            # Log when we actually start training the networks
            logger.log_text({"Net train started at": epoch}, -1)
            net_train_started = True

        net.train()
        # collect statistics for distribution of data across fn training batches, i.e. 
        # Over all training batches, compute avg and std of:
        # - pretrain success in a batch - cotrain success in a batch - overall success in a batch
        # - all three above for failures
        per_batch_stats = {
            "pretrain_success": np.empty(0),
            "pretrain_failure": np.empty(0),
            "cotrain_success": np.empty(0),
            "cotrain_failure": np.empty(0),
            "overall_success": np.empty(0),
            "overall_failure": np.empty(0),
        }
        for i in range(hyperparams['cotrain_fn_max_epochs']):
            mean_fitness_loss, _, fitness_gts_batches, _, dataset_types_batches, _= \
                optimize_network(
                    data_loader=get_fn_loader(),
                    net=net,
                    fitness_criterion=fitness_criterion,
                    logger=logger,
                    tqdm_prefix=f'Epoch {i + 1} |',
                    get_fitness_loss=True,
                    optimize_fitness_net=True,
                    get_generator_loss=False,
                    optimize_generator_net=False)
            for fitness_gts_batch, dataset_types_batch in zip(fitness_gts_batches, dataset_types_batches):
                pretrain_indices = dataset_types_batch == GraspDatasetType.PRETRAIN
                cotrain_indices = dataset_types_batch == GraspDatasetType.COTRAIN
                success_indices = fitness_gts_batch == 1
                failure_indices = fitness_gts_batch == 0
                total_num_indices = len(fitness_gts_batch)
                per_batch_stats["pretrain_success"] = np.append(
                    per_batch_stats["pretrain_success"],
                    np.sum(pretrain_indices & success_indices) / total_num_indices
                    if total_num_indices != 0 else 0
                )
                per_batch_stats["pretrain_failure"] = np.append(
                    per_batch_stats["pretrain_failure"],
                    np.sum(pretrain_indices & failure_indices) / total_num_indices
                    if total_num_indices != 0 else 0
                )
                per_batch_stats["cotrain_success"] = np.append(
                    per_batch_stats["cotrain_success"],
                    np.sum(cotrain_indices & success_indices) / total_num_indices
                    if total_num_indices != 0 else 0
                )
                per_batch_stats["cotrain_failure"] = np.append(
                    per_batch_stats["cotrain_failure"],
                    np.sum(cotrain_indices & failure_indices) / total_num_indices
                    if total_num_indices != 0 else 0
                )
                per_batch_stats["overall_success"] = np.append(
                    per_batch_stats["overall_success"],
                    np.mean(success_indices)
                    if total_num_indices != 0 else 0
                )
                per_batch_stats["overall_failure"] = np.append(
                    per_batch_stats["overall_failure"],
                    np.mean(failure_indices)
                    if total_num_indices != 0 else 0
                )
            if i >= hyperparams['cotrain_fn_min_epochs'] - 1\
                    and mean_fitness_loss < hyperparams['cotrain_target_fn_loss']:
                break
    
        mean_per_batch_stats = dict()
        std_per_batch_stats = dict()
        for k, v in per_batch_stats.items():
            mean_per_batch_stats[k] = v.mean()
            std_per_batch_stats[k] = v.std()
        logger.log_scalars({
            "FN_Mean_Batch_Dist": mean_per_batch_stats,
            "FN_Std_Batch_Dist": std_per_batch_stats
        }, epoch)

        # ============================================
        # Add statistics after training Fitness network
        # ============================================
        # - fn performance on cotrain dataset
        after_fn_losses_dict, after_fn_preds_dict =  collect_all_fn_performance("after")
        # - plot both before and after performance together
        after_fn_losses_dict.update(before_fn_losses_dict)
        logger.log_scalars({"FN_MAE_GS": after_fn_losses_dict}, epoch)
        after_fn_preds_dict.update(before_fn_preds_dict)
        logger.log_histogram(after_fn_preds_dict, epoch)

        # ============================================
        # Train generator network
        # ============================================
        for i in range(hyperparams['cotrain_gn_max_epochs']):
            _, mean_generator_loss, _, _, _, _ = \
                optimize_network(
                    data_loader=get_gn_loader(),
                    net=net,
                    fitness_criterion=fitness_criterion,
                    logger=logger,
                    tqdm_prefix=f'Epoch {i + 1} |',
                    get_fitness_loss=False,
                    optimize_fitness_net=False,
                    get_generator_loss=True,
                    optimize_generator_net=True,
                    max_optimization_steps=hyperparams['cotrain_gn_max_opt_steps_per_epoch'])

            if i >= hyperparams['cotrain_gn_min_epochs'] - 1\
                    and mean_generator_loss < hyperparams['cotrain_target_gn_loss']:
                break

        net.save(epoch, '{}/ckpt_{:03d}.pth'.format(logger.logdir, epoch))
        print('Epoch {} took {:.01f} seconds.'.format(
            epoch,
            float(time() - start_epoch_time)))


def pretrain(
        args,
        config,
        seed=0):
    """
    more reading: https://pytorch.org/docs/stable/distributed.html
    """
    seed_all(seed)
    rank = 0
    gpus = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpus)
    assert num_gpus == 1
    device = gpus[rank]
    print("[{}] Using GPU {} out of {} GPUS ({} available)".format(
        rank, device, device_count(), num_gpus))
    set_device(device)
    net, hyperparams, get_train_loader, get_val_loader, logger =\
        setup_pretrain(args, config)
    per_gpu_batch_size = int(hyperparams['pretrain_batch_size'] / num_gpus)
    fitness_criterion = BCELoss()

    if 'fn_update_steps' not in net.stats:
        net.stats['fn_update_steps'] = 0
    if 'gn_update_steps' not in net.stats:
        net.stats['gn_update_steps'] = 0

    for epoch in range(net.stats['epochs'],
                       hyperparams['pretrain_num_epochs']):
        print("=" * 25 + f"EPOCH {epoch}" + "=" * 25)
        # 1. Train
        net.train()
        mean_fitness_loss, _, _, _, _, _ = \
            optimize_network(
                data_loader=get_train_loader(per_gpu_batch_size),
                net=net,
                fitness_criterion=fitness_criterion,
                logger=logger,
                tqdm_prefix='Train |',
                optimize_fitness_net=True,
                optimize_generator_net=False,
                get_generator_loss=False)
        mean_fitness_loss /= num_gpus
        logger.log(
            data={
                'Train/Fitness_Loss': mean_fitness_loss,
            },
            step=epoch)

        # 2. Validate
        with no_grad():
            net.eval()
            mean_fitness_loss, _, _, _, _, _ = \
                optimize_network(
                    data_loader=get_val_loader(per_gpu_batch_size),
                    net=net,
                    fitness_criterion=fitness_criterion,
                    logger=None,
                    tqdm_prefix='Validation |',
                    get_fitness_loss=True,
                    optimize_fitness_net=False,
                    optimize_generator_net=False,
                    get_generator_loss=False,
                )
        mean_fitness_loss /= num_gpus
        print("Average validation | F: {:.04f} ".format(mean_fitness_loss))
        logger.log(
            data={
                'Validation/Fitness_Loss': mean_fitness_loss,
            },
            step=epoch)

        net.save(epoch, '{}pretrain_{:03d}.pth'.format(
            logger.logdir, epoch))


def pretrain_gn(
        args,
        config,
        seed=0):
    """
    more reading: https://pytorch.org/docs/stable/distributed.html
    """
    seed_all(seed)
    rank = 0
    gpus = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpus)
    assert num_gpus == 1
    device = gpus[rank]
    print("[{}] Using GPU {} out of {} GPUS ({} available)".format(
        rank, device, device_count(), num_gpus))
    set_device(device)
    net, hyperparams, get_train_loader, get_val_loader, logger =\
        setup_pretrain(args, config)
    per_gpu_batch_size = int(hyperparams['pretrain_batch_size'] / num_gpus)

    if 'gn_update_steps' not in net.stats:
        net.stats['gn_update_steps'] = 0
    generator_criterion = MSELoss()
    for epoch in range(net.stats['epochs'],
                       hyperparams['pretrain_num_epochs']):
        print("=" * 25 + f"EPOCH {epoch}" + "=" * 25)
        # 1. Train
        net.train()
        mean_generator_loss_train = \
            optimize_generator(
                data_loader=get_train_loader(per_gpu_batch_size),
                net=net,
                generator_criterion=generator_criterion,
                logger=logger,
                tqdm_prefix='Train |',
                optimize=True)
        mean_generator_loss_train = mean_generator_loss_train.cpu().item() / num_gpus
        print(f"Train | G: {mean_generator_loss_train:.04f}")
        logger.log(
            data={
                'Train/Generator_Loss': mean_generator_loss_train,
            },
            step=epoch)

        # 2. Validate
        with no_grad():
            net.eval()
            mean_generator_loss_val = \
                optimize_generator(
                    data_loader=get_val_loader(per_gpu_batch_size),
                    net=net,
                    generator_criterion=generator_criterion,
                    logger=logger,
                    tqdm_prefix='Val |',
                    optimize=False)
        mean_generator_loss_val = mean_generator_loss_val.cpu().item() / num_gpus
        print("Validation | G: {:.04f} ".format(mean_generator_loss_val))
        logger.log(
            data={
                'Validation/Generator_Loss': mean_generator_loss_val,
            },
            step=epoch)

        net.save(epoch, '{}imprint_pretrain_gn_{:03d}.pth'.format(
            logger.logdir, epoch))


def train_vae(net, hyperparams: dict, train_loader, val_loader,
              logger: Logger):
    vae_criternion = MSELoss()
    optimizer = net.vae_optimizer
    for epoch in range(net.stats['epochs'], hyperparams['vae_num_epochs']):
        print("=" * 10 + f"EPOCH {epoch}" + "=" * 10)
        with tqdm(train_loader, dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch = batch.to(net.device, non_blocking=True)
                vae_loss = vae_criternion(
                    net.encode_decode(batch),
                    batch)
                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()

                logger.log(
                    data={
                        'Training/VAE_Loss': vae_loss.item()
                    },
                    step=net.stats['update_steps'])
                pbar.set_description('Train Loss: {:.04f}'.format(
                    vae_loss.item()
                ))
                net.stats['update_steps'] += 1

        with tqdm(val_loader, dynamic_ncols=True)\
                as pbar, no_grad():
            val_losses = []
            for batch in pbar:
                batch = batch.to(net.device)
                vae_loss = vae_criternion(
                    net.encode_decode(batch),
                    batch)
                val_losses.append(vae_loss.item())
                pbar.set_description('Validation Loss: {:.04f}'.format(
                    vae_loss.item()
                ))
            logger.log(
                data={
                    'Validation/VAE_Loss': np.mean(val_losses)
                },
                step=epoch)
            print(f'\rValidation Loss:{np.mean(val_losses)}')
        checkpoint_path = '{}vae_{:03d}.pth'.format(logger.logdir, epoch)
        net.save(epoch, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
