from argparse import ArgumentParser
from environment import GraspSimulationEnv
from distribute import Pool
from utils import dicts_get, load_config, load_evaluate_config
from learning import ObjectDataset
import numpy as np
import ray
import os
from tqdm import tqdm
from learning.finger_generator import get_finger_generator, ImprintFingerGenerator
from shutil import copyfile
from os.path import splitext
from json import dump
import pickle

def acc_results(results):
    metrics = {
        'base_connected': np.empty(0),
        'created_grippers_failed': np.empty(0),
        'single_connected_component': np.empty(0),
        'grasp_object_path': []
    }
    for key in results[0]['score'].keys():
        metrics[key] = np.empty(0)
    for r in results:
        if r["score"] is None:
            print("Score is none. Ignoring ...")
            continue
        metrics['base_connected'] = np.concatenate(
            (metrics['base_connected'],
                [r['base_connected']]))
        metrics['created_grippers_failed'] = np.concatenate(
            (metrics['created_grippers_failed'],
                [r['created_grippers_failed']]))
        metrics['single_connected_component'] = np.concatenate(
            (metrics["single_connected_component"],
                [r["single_connected_component"]]))
        metrics['grasp_object_path'].append(r['grasp_object_path'])
        for key, r_val in r["score"].items():
            metrics[key] = np.concatenate((metrics[key], [r_val]))
    for key, val in metrics.items():
        if key is not 'grasp_object_path':
            metrics[key] = metrics[key].astype(float)
    return metrics


def print_results(metrics, name=""):
    print(f"Results summary {name}:")
    for key, val in metrics.items():
        if key != 'grasp_object_path':
            print(f"{key}: {np.mean(np.array(val, dtype=float)):.4f}")


if __name__ == '__main__':
    parser = ArgumentParser('Generator fingers evaluater')
    parser.add_argument("--evaluate_config",
                        help="path to evaluate config file",
                        required=True,
                        type=str)
    parser.add_argument('--objects',
                        help='path to shapenet root',
                        type=str,
                        required=True)
    parser.add_argument('--config',
                        help='path to JSON config file',
                        type=str,
                        default='configs/default.json')
    parser.add_argument('--name',
                        help='path to directory to save results',
                        type=str, required=True)
    parser.add_argument('--num_processes',
                        help='number of environment processes',
                        type=int,
                        default=32)
    parser.add_argument('--gui', action='store_true',
                        default=False, help='Run headless or render')
    parser.add_argument('--objects_bs',
                        help='objects batch size',
                        type=int,
                        default=128)
    args = parser.parse_args()

    # load environment
    ray.init()
    config = load_config(args.config)
    env_pool = Pool([GraspSimulationEnv.remote(
        config=config,
        gui=args.gui
    ) for _ in range(args.num_processes)])

    # object dataset
    obj_dataset = ObjectDataset(
        directory_path=args.objects,
        batch_size=args.objects_bs,
        grasp_object_file="graspobject_test_all.txt",
    )
    grippers_directory = args.name + '/'
    assert not os.path.exists(grippers_directory)
    os.makedirs(grippers_directory)

    # evaluations
    evaluate_config = load_evaluate_config(args.evaluate_config)

    # dump config and args
    print("logging to ", grippers_directory)
    dump(config, open(grippers_directory + 'config.json', 'w'), indent=4)
    dump(evaluate_config, open(grippers_directory + 'evaluate_config.json', 'w'), indent=4)
    pickle.dump(args,open(grippers_directory + 'args.pkl','wb'))

    finger_generators = []
    for finger_generator_config in evaluate_config:
        if not finger_generator_config["evaluate"]:  # TODO: Do it more neatly
            continue
        results = []
        finger_generator = get_finger_generator(finger_generator_config)
        print(finger_generator)
        obj_loader = iter(obj_dataset.loader)
        for i, grasp_objects in enumerate(tqdm(obj_loader, smoothing=0.01, dynamic_ncols=True, desc=finger_generator_config["desc"])):
            retval = finger_generator.create_fingers(
                grasp_objects)
            gripper_output_paths = ['{}/{:06}'.format(
                grippers_directory, i + len(results))
                for i in range(len(grasp_objects))]
            if type(finger_generator) == ImprintFingerGenerator:
                left_fingers, right_fingers, gripper_urdf_paths = retval
                for gripper_urdf_path, target_gripper_urdf_path in \
                        zip(gripper_urdf_paths, gripper_output_paths):
                    # copy collision meshes over
                    prefix = splitext(gripper_urdf_path)[0]
                    copyfile(prefix + '_right_collision.obj',
                             target_gripper_urdf_path + '_right_collision.obj')
                    copyfile(prefix + '_left_collision.obj',
                             target_gripper_urdf_path + '_left_collision.obj')
            else:
                left_fingers, right_fingers = retval
            grasp_results = env_pool.map(
                exec_fn=lambda env, args:
                env.compute_finger_grasp_score.remote(*args),
                iterable=zip(
                    left_fingers,
                    right_fingers,
                    dicts_get(grasp_objects, 'urdf_path'),
                    gripper_output_paths,
                    [False] * len(gripper_output_paths)))
            results.extend(grasp_results)
            metrics = acc_results(results)
            print_results(metrics, name=finger_generator_config["desc"])
            save_path = os.path.join(grippers_directory, "val_results.npz")
            np.savez(save_path, **metrics)
            print(f"saved results at {save_path}")
