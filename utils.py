from argparse import ArgumentParser
from copy import deepcopy
from json import load, dump
from os.path import dirname, basename, exists, splitext, isdir
from os import makedirs
import os
from tensorboardX import SummaryWriter
from environment import (
    GraspSimulationEnv,
    GraspObjectGenerationEnv,
    ImprintGenerationEnv
)
from learning import (
    ObjectDataset,
    ImprintObjectDataset,
    GraspDataset,
    ConcatGraspDataset,
    BalancedGraspDataset,
    GripperDesigner,
    get_loader,
    grasp_dataset_concat_collate_fn,
    VAEDatasetHDF,
    GraspDatasetType, 
)
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import ConcatDataset, DataLoader
from git import Repo
from pathlib import Path
from signal import signal, SIGINT
import socket
from contextlib import closing
import numpy as np
import random
import torch
import pickle


def seed_all(seed=0):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    parser = ArgumentParser('Fit2Form')
    parser.add_argument('--name',
                        help='name of experiment',
                        type=str,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=0)
    parser.add_argument('--config',
                        help='path to JSON config file',
                        type=str,
                        default="configs/default.json")
    parser.add_argument('--objects',
                        help='path to grasp objects directory',
                        type=str,
                        default="data/ShapeNetCore.v2")
    parser.add_argument("--ae_checkpoint_path", type=str, default=None)
    parser.add_argument("--gn_checkpoint_path", type=str, default=None)
    parser.add_argument("--fn_checkpoint_path", type=str, default=None)
    parser.add_argument('--grasp_dataset',
                        help='path to HDF5 grasp dataset',
                        type=str,
                        default=None)
    parser.add_argument('--train',
                        help='path to train.txt file',
                        type=str,
                        default=None)
    parser.add_argument('--val',
                        help='path to val.txt file',
                        type=str,
                        default=None)
    parser.add_argument("--test",
                        help="path to test.txt file",
                        default=None
    )
    parser.add_argument('--split',
                        help='choose which dataset split to use',
                        choices=['train', 'val'],
                        default=None
                        )
    parser.add_argument("--shapenet_train_hdf", help="path to processed shapenet train hdf file", default=None)
    parser.add_argument("--shapenet_val_hdf", help="path to processed shapenet val hdf file", default=None)
    parser.add_argument("--imprint_train_hdf", help="path to processed imprint train hdf file", default=None)
    parser.add_argument("--imprint_val_hdf", help="path to processed imprint val hdf file", default=None)
    parser.add_argument('--mode',
                        choices=[
                            # 1. collision mesh for .obj files
                            'collision_mesh',
                            # 2. create urdf for all visual and collision mesh
                            'urdf',
                            # 3. create OBJ mesh, collision mesh, URDF, and
                            # TSDF of random, stable orientation of object
                            'grasp_objects',
                            # 4. generate chopped and rescaled shapenet grasp dataset
                            'pretrain_dataset',
                            # 5. generate imprint baseline for grasp objects
                            'imprint_baseline',
                            #     # 6. generate imprint grasp dataset
                            'pretrain_imprint_dataset',
                            'vae',
                            'pretrain',
                            'pretrain_gn',  # pretrain gn to regress imprint fingers
                            'cotrain'
                        ],
                        default='cotrain')
    parser.add_argument('--gpus',
                        type=str,
                        default='0',
                        help='comma-separated GPU device ids')
    parser.add_argument('--num_processes',
                        help='number of environment processes',
                        type=int,
                        default=32)
    parser.add_argument('--num_grasp_objects',
                        help='number of grasp_objects to generate',
                        type=int,
                        default=200000)
    parser.add_argument('--num_pretrain_dataset',
                        help='number of grasp simuations to do for pretrain dataset',
                        type=int,
                        default=1000000)

    parser.add_argument('--gui', action='store_true',
                        default=False, help='Run headless or render')
    args = parser.parse_args()
    return args


def merge(a, b):
    # Merge two dictionaries
    if isinstance(b, dict) and isinstance(a, dict):
        a_and_b = a.keys() & b.keys()
        every_key = a.keys() | b.keys()
        return {k: merge(a[k], b[k]) if k in a_and_b else
                deepcopy(a[k] if k in a else b[k]) for k in every_key}
    return deepcopy(b)


def load_config(path, merge_with_default=True):
    print(path)
    base_dirname = basename(dirname(path))
    merge_with_default = (base_dirname == 'configs')

    if splitext(basename(path))[0] != 'default'\
            and merge_with_default:
        config = load(open(dirname(path) + '/default.json'))
        additional_config = load(open(path))
        config = merge(config, additional_config)
    else:
        config = load(open(path))
    return config


def load_evaluate_config(path):
    print(f"Loading evaluation config from path: {path}")
    config = load(open(path))
    for finger_generator_config in config["evaluations"]:
        if not finger_generator_config["evaluate"]:
            continue
        if "train_config_path" in finger_generator_config:
            train_config_path = finger_generator_config["train_config_path"]
        elif len(finger_generator_config['checkpoint_dict']):
            train_config_path = Path(list(finger_generator_config["checkpoint_dict"].values())[
                                     0]).parent.joinpath("config.json")
        else:
            continue
        finger_generator_config["train_config"] = load_config(
            train_config_path, merge_with_default=False)
    return config["evaluations"]


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def dicts_get(dicts, key):
    return [item[key] for item in dicts
            if item is not None and key in item]


def get_experiment_directory(args):
    if args.name is None and args.load is not None:
        args.name = basename(dirname(args.load))
    experiment_directory = f'runs/{args.name}/'
    if not exists(experiment_directory):
        makedirs(experiment_directory)
    return experiment_directory


class Logger:
    def __init__(self, args, config):
        self.logdir = get_experiment_directory(args)
        print("[Logger] logging to ", self.logdir)
        dump(config, open(self.logdir + 'config.json', 'w'), indent=4)
        pickle.dump(args,open(self.logdir + 'args.pkl','wb'))
        self.writer = SummaryWriter(logdir=self.logdir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def log_histogram(self, data, step):
        for key in data:
            self.writer.add_histogram(key, data[key], step)
    
    def log_text(self, data, step):
        for key in data:
            self.writer.add_text(key, str(data[key]), step)

    def log_histogram(self, data, step):
        for key in data:
            self.writer.add_histogram(key, data[key], step)

    def log_scalars(self, data, step):  # For plotting multiple plots in same graph
        for key in data:
            self.writer.add_scalars(key, data[key], step)


def exit_handler():
    print("Gracefully terminating")
    exit(0)


def setup_pretrain_dataset_generation(
        args,
        config,
        is_imprint=False,
        batch_size=512):
    if args.name is None:
        print("Please supply directory to store pretrain dataset with --name")
        exit()
    if args.name[-1] != '/':
        args.name += '/'
    if args.objects is None:
        print("Please provide path to shapenet root with --objects")
        exit()
    if args.split is None:
        print("Specify which dataset split to use with --split")
        exit()
    split_map = {
        "train": args.train,
        "val": args.val,
        "test": args.test
    }
    if split_map[args.split] is None:
        print(f"Please specify category split file with --{split_map[args.split]} corresponding to args.split {args.split}")
        exit()
    category_file = split_map[args.split]

    if not exists(args.name):
        makedirs(args.name)
    GraspObjectDataset = ImprintObjectDataset\
        if is_imprint\
        else ObjectDataset
    return (
        [GraspSimulationEnv.remote(
            config=config,
            gui=args.gui)
            for _ in range(args.num_processes)],
        GraspObjectDataset(
            directory_path=args.objects,
            batch_size=batch_size,
            category_file=category_file,
        ),
        GraspDataset(
            directory_path=args.name,
            batch_size=batch_size,
            suffix = args.split
            ),
        config['environment']['tsdf_voxel_size'],
        args.num_pretrain_dataset,
    )

def setup_imprint_generation(args, config):
    if args.objects is None:
        print("Please supply path to shapenet root with --objects")
        exit()
    return (
        [ImprintGenerationEnv.remote(
            config=config,
            gui=args.gui)
            for _ in range(args.num_processes)],
        ObjectDataset(
            directory_path=args.objects,
            batch_size=args.num_processes)
    )


def setup_network(args, hyperparameters):
    device = 'cuda' if is_cuda_available() else 'cpu'
    net = GripperDesigner(
        embedding_dim=hyperparameters['embedding_dim'],
        designer_weight_decay=hyperparameters['designer_weight_decay'],
        designer_lr=hyperparameters['designer_lr'],
        vae_weight_decay=hyperparameters['vae_weight_decay'],
        vae_lr=hyperparameters['vae_lr'],
        device=device,
        fitness_class_str=hyperparameters["fitness_class"]
        if "fitness_class" in hyperparameters else None,
        generator_class_str=hyperparameters["generator_class"]
        if "generator_class" in hyperparameters else None,
        optimize_objectives=hyperparameters['optimize_objectives'],
        use_1_robustness=hyperparameters["use_1_robustness"]
        if "use_1_robustness" in hyperparameters else None,
        checkpoint_dict={
            "ae_checkpoint_path": args.ae_checkpoint_path,
            "gn_checkpoint_path": args.fn_checkpoint_path,
            "fn_checkpoint_path": args.gn_checkpoint_path,
        }
    )
    return net


def setup_pretrain_dataset(hdf5_paths, hyperparameters, dataset_class):
    assert len(hdf5_paths)
    datasets = []
    for dataset_path in hdf5_paths:
        dataset_path = str(dataset_path)
        datasets.append(
            dataset_class(
                dataset_path=dataset_path,
                batch_size=hyperparameters['pretrain_batch_size'],
                optimize_objectives=hyperparameters['optimize_objectives'],
                use_1_robustness=hyperparameters['use_1_robustness'] if 'use_1_robustness' in hyperparameters else False,
                ))
    return ConcatGraspDataset(datasets)


def get_pretrain_hdf_paths(args):
    if args.imprint_train_hdf is None or args.imprint_val_hdf is None:
        print("Supply paths to imprint_train_hdf(imprint_val_hdf) with --imprint_train_hdf(--imprint_val_hdf)")
        exit() 
    train_hdf5_paths = [args.imprint_train_hdf]
    val_hdf5_paths=[args.imprint_val_hdf]
    if args.mode == "pretrain":
        if args.shapenet_train_hdf is None or args.shapenet_val_hdf is None:
            print("Supply paths to shapenet_train_hdf(shapenet_val_hdf) with --shapenet_train_hdf(--shapenet_val_hdf)")
            exit() 
        train_hdf5_paths.append(args.shapenet_train_hdf)
        val_hdf5_paths.append(args.shapenet_val_hdf)
    return train_hdf5_paths, val_hdf5_paths

def setup_pretrain_datasets(args, hyperparameters):
    """
    This function abstracts away from the dataset
    and only returns a function to get a dataloader.
    This prevents accidental modifications to the
    pretrain dataset
    """
    dataset_class = BalancedGraspDataset\
        if hyperparameters['balanced_sampling']\
        else GraspDataset
    collate_fn = grasp_dataset_concat_collate_fn\
        if hyperparameters['balanced_sampling']\
        else None
    

    train_hdf5_paths, val_hdf5_paths = get_pretrain_hdf_paths(args)
    train_dataset = setup_pretrain_dataset(
        hdf5_paths=train_hdf5_paths,
        hyperparameters=hyperparameters,
        dataset_class=dataset_class
    )

    val_dataset = setup_pretrain_dataset(
        hdf5_paths=val_hdf5_paths,
        hyperparameters=hyperparameters,
        dataset_class=dataset_class
    )

    def get_train_loader(batch_size=hyperparameters['pretrain_batch_size']):
        if hyperparameters['balanced_sampling']:
            batch_size = int(batch_size / 2)
        return get_loader(dataset=train_dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          distributed=False)

    def get_val_loader(batch_size=hyperparameters['pretrain_batch_size']):
        if hyperparameters['balanced_sampling']:
            batch_size = int(batch_size / 2)
        return get_loader(dataset=val_dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          distributed=False)
    return get_train_loader, get_val_loader


def setup_pretrain(args, config):
    if args.name is None:
        print("Supply experiment name with --name")
    elif args.config is None:
        print("Supply path to JSON config file with --config")
    elif args.objects is None:
        print("Supply root of shapenet with --objects")
    hyperparameters = config['training']['hyperparameters']
    net = setup_network(args, hyperparameters)
    get_train_loader, get_val_loader = \
        setup_pretrain_datasets(args, hyperparameters)

    return net, hyperparameters,\
        get_train_loader, get_val_loader, Logger(args, config)

def setup_train_vae(args, config):
    if args.name is None:
        print("Supply experiment name with --name")
        exit() 
    if args.config is None:
        print("Supply path to JSON config file with --config")
        exit() 
    if args.objects is None:
        print("Supply root of shapenet with --objects")
        exit() 
    if args.shapenet_train_hdf is None or args.shapenet_val_hdf is None:
        print("Supply paths to shapenet_train_hdf(shapenet_val_hdf) with --shapenet_train_hdf(--shapenet_val_hdf)")
        exit() 
    if args.imprint_train_hdf is None or args.imprint_val_hdf is None:
        print("Supply paths to imprint_train_hdf(imprint_val_hdf) with --imprint_train_hdf(--imprint_val_hdf)")
        exit() 
    hyperparameters = config['training']['hyperparameters']
    logger = Logger(args, config)

    # add logic for making train and val loader
    train_vae_datasets = list()
    val_vae_datasets = list()

    # imprint
    train_vae_datasets.append(VAEDatasetHDF(dataset_path=args.imprint_train_hdf))
    val_vae_datasets.append(VAEDatasetHDF(dataset_path=args.imprint_val_hdf))
    # shapenet
    train_vae_datasets.append(VAEDatasetHDF(dataset_path=args.shapenet_train_hdf))
    val_vae_datasets.append(VAEDatasetHDF(dataset_path=args.shapenet_val_hdf))

    # concatenate datasets
    train_vae_dataset = ConcatDataset(train_vae_datasets)
    val_vae_dataset = ConcatDataset(val_vae_datasets)
    train_loader = DataLoader(train_vae_dataset,
                              batch_size=hyperparameters['vae_batch_size'],
                              num_workers=6,
                              shuffle=True)
    val_loader = DataLoader(val_vae_dataset,
                            batch_size=hyperparameters['vae_batch_size'],
                            num_workers=6,
                            shuffle=True)
    return setup_network(args, hyperparameters),\
        hyperparameters, \
        train_loader, \
        val_loader, \
        logger


def setup(args, config):
    if args.mode == 'pretrain_dataset':
        return setup_pretrain_dataset_generation(args, config)
    elif args.mode == 'pretrain_imprint_dataset':
        return setup_pretrain_dataset_generation(args, config, is_imprint=True)
    elif args.mode == 'imprint_baseline':
        return setup_imprint_generation(args, config)
    if args.mode == 'cotrain':
        Env = GraspSimulationEnv
    elif args.mode == 'grasp_objects':
        Env = GraspObjectGenerationEnv
    signal(SIGINT, lambda sig, frame: exit_handler())
    envs = [Env.remote(
        config=config,
        gui=args.gui)
        for _ in range(args.num_processes)]
    if args.mode == 'grasp_objects':
        return envs, args.objects, args.num_grasp_objects

    logger = Logger(args, config)
    hyperparameters = config['training']['hyperparameters']
    net = setup_network(args, hyperparameters)
    train_hdf_paths, _ = get_pretrain_hdf_paths(args)
    pretrain_training_dataset = setup_pretrain_dataset(
        hdf5_paths=train_hdf_paths,
        hyperparameters=hyperparameters,
        dataset_class=GraspDataset)

    return (
        envs,
        net,
        hyperparameters,
        ObjectDataset(
            directory_path=args.objects,
            batch_size=hyperparameters['cotrain_batch_size']),
        GraspDataset(
            directory_path=logger.logdir,
            batch_size=hyperparameters['cotrain_batch_size'],
            use_latest_points=hyperparameters['use_latest_points'],
            optimize_objectives=hyperparameters['optimize_objectives'],
            use_1_robustness=hyperparameters['use_1_robustness'] if 'use_1_robustness' in hyperparameters else False,
            grasp_dataset_type=GraspDatasetType.COTRAIN),
        pretrain_training_dataset,
        logger
    )
