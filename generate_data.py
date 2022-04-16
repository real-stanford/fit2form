from random import shuffle, seed
from pathlib import Path
from math import ceil
from tqdm import tqdm
from numpy import array
from utils import dicts_get
from environment.meshUtils import create_shapenet_gripper
from os.path import exists, splitext
from os import mkdir
from distribute import Pool
from time import time
from torch import manual_seed
from torch.utils.data import DataLoader
import ray


def create_grasp_objects(envs, root_dir, total=200000):
    """
    For each input URDF, create:
     - OBJ visual and collision mesh
     - URDF with OBJ visual and collision mesh
     - TSDF
    """
    seed(time())
    env_pool = Pool(envs)
    urdf_paths = [str(path)
                  for path in Path(root_dir).rglob('*normalized.urdf')]
    shuffle(urdf_paths)
    urdf_paths = urdf_paths * ceil(total * 1.5 / len(urdf_paths))
    grasp_object_paths = env_pool.map_unordered(
        exec_fn=lambda env, urdf_path:
        env.create_grasp_object.remote(urdf_path),
        iterable=urdf_paths,
        desc='Creating grasp objects')

    print('Created {} grasp objects in {}'.format(
        len(grasp_object_paths),
        root_dir))


def generate_pretrain_data(envs,
                           grasp_object_dataset,
                           grasp_dataset,
                           voxel_size,
                           total=1000000):
    manual_seed(time())
    env_pool = Pool(envs)
    output_dir = f'{grasp_dataset.directory_path}grippers/'
    if not exists(output_dir):
        mkdir(output_dir)
    pbar = tqdm(total=total, desc='Generating pretrain dataset',
                dynamic_ncols=True)
    pbar.update(len(grasp_dataset))
    async_create_shapenet_gripper = ray.remote(create_shapenet_gripper)
    while True:
        pbar.set_description('Sampling grasp objects')
        grasp_objects = grasp_object_dataset.sample()
        pbar.set_description('Sampling left fingers')
        left_fingers = grasp_object_dataset.sample()
        pbar.set_description('Sampling right fingers')
        right_fingers = grasp_object_dataset.sample()

        pbar.set_description('Creating grippers')
        grippers = ray.get([
            async_create_shapenet_gripper.remote(
                left_finger_tsdf=left_finger_tsdf,
                right_finger_tsdf=right_finger_tsdf,
                output_prefix=output_prefix,
                voxel_size=voxel_size)
            for left_finger_tsdf, right_finger_tsdf, output_prefix in
            zip(
                dicts_get(left_fingers, 'tsdf'),
                dicts_get(right_fingers, 'tsdf'),
                ['{}{:09d}'.format(
                    output_dir, len(grasp_dataset) + i)
                 for i in range(len(grasp_objects))],
            )])
        if len(list(filter(None, grippers))) == 0:
            continue
        # filter out invalid grippers
        grasp_objects, grippers = zip(
            *((grasp_object, gripper)
              for grasp_object, gripper in
              zip(grasp_objects, grippers)
              if gripper))
        left_finger_tsdfs = dicts_get(grippers, 'left_finger_tsdf')
        right_finger_tsdfs = dicts_get(grippers, 'right_finger_tsdf')
        gripper_urdf_paths = dicts_get(grippers, 'gripper_urdf_path')
        grasp_object_tsdf_paths = dicts_get(grasp_objects, "tsdf_path")
        grasp_object_urdfs = dicts_get(grasp_objects, 'urdf_path')
        pbar.set_description('Simulating grasps')
        grasp_results = env_pool.map(
            exec_fn=lambda env, args:
            env.simulate_grasp.remote(*args),
            iterable=zip(
                grasp_object_urdfs,
                gripper_urdf_paths,
                left_finger_tsdfs,
                right_finger_tsdfs))
        grasp_data = [
            {
                'grasp_object_tsdf_path': grasp_object_tsdf_path,
                'left_finger_tsdf': left_finger_tsdf,
                'right_finger_tsdf': right_finger_tsdf,
                'grasp_result': array(list(grasp_score.values()))
            }
            for
            grasp_object_tsdf_path,
            left_finger_tsdf,
            right_finger_tsdf,
            grasp_score
            in zip(grasp_object_tsdf_paths,
                   left_finger_tsdfs,
                   right_finger_tsdfs,
                   grasp_results)
            if grasp_score is not None]
        grasp_dataset.extend(grasp_data, log=False)
        pbar.update(len(grasp_data))
        if len(grasp_dataset) > total:
            exit()


def update_grasp_dataset(
        grasp_results,
        left_finger_tsdf_paths,
        right_finger_tsdf_paths,
        grasp_object_tsdf_paths,
        grasp_dataset):
    grasp_data = [
        {
            'grasp_object_tsdf_path': grasp_object_tsdf_path,
            'left_finger_tsdf_path': left_finger_tsdf_path,
            'right_finger_tsdf_path': right_finger_tsdf_path,
            'grasp_result': array(list(grasp_score.values()))
        }
        for
        grasp_object_tsdf_path,
        left_finger_tsdf_path,
        right_finger_tsdf_path,
        grasp_score
        in zip(grasp_object_tsdf_paths,
               left_finger_tsdf_paths,
               right_finger_tsdf_paths,
               grasp_results)
        if grasp_score is not None]
    grasp_dataset.extend(grasp_data, log=False)


def generate_pretrain_imprint_data(
    envs,
    grasp_object_dataset,
    grasp_dataset,
    voxel_size=None,
):
    env_pool = Pool(envs)
    output_dir = f"{grasp_dataset.directory_path}grippers_imprint/"
    if not exists(output_dir):
        mkdir(output_dir)
    pbar = tqdm(total=len(grasp_object_dataset),
                desc='Generating pretrain imprint dataset')
    loader = DataLoader(grasp_object_dataset, batch_size=grasp_object_dataset.batch_size,
                        shuffle=False, collate_fn=lambda batch: batch)
    for grasp_objects in loader:
        grasp_object_urdfs = dicts_get(grasp_objects, 'urdf_path')
        gripper_urdfs = [splitext(grasp_object_urdf)[
            0] + "_imprint.urdf" for grasp_object_urdf in grasp_object_urdfs]
        left_finger_tsdf_paths = [splitext(grasp_object_urdf)[
            0] + "_imprint_left_tsdf.npy" for grasp_object_urdf in grasp_object_urdfs]
        right_finger_tsdf_paths = [splitext(grasp_object_urdf)[
            0] + "_imprint_right_tsdf.npy" for grasp_object_urdf in grasp_object_urdfs]
        grasp_object_tsdf_paths = dicts_get(grasp_objects, "tsdf_path")
        update_grasp_dataset(
            grasp_results=env_pool.map(
                exec_fn=lambda env, args:
                env.simulate_grasp.remote(*args),
                iterable=zip(grasp_object_urdfs, gripper_urdfs)),
            left_finger_tsdf_paths=left_finger_tsdf_paths,
            right_finger_tsdf_paths=right_finger_tsdf_paths,
            grasp_object_tsdf_paths=grasp_object_tsdf_paths,
            grasp_dataset=grasp_dataset
        )
        pbar.update(len(grasp_object_urdfs))


def generate_imprints(envs, grasp_object_dataset):
    env_pool = Pool(envs)
    grasp_object_dataset.get_urdf_path_only = True

    imprint_urdfs = env_pool.map_unordered(
        exec_fn=lambda env, grasp_object_urdf_path:
        env.create_imprint_gripper_fingers.remote(
            grasp_object_urdf_path=grasp_object_urdf_path),
        iterable=grasp_object_dataset,
        desc='Creating imprint baseline'
    )
    imprint_urdfs = [urdf for urdf in imprint_urdfs if urdf is not None]
    print("Generated {}/{} imprints.".format(
        len(imprint_urdfs), len(grasp_object_dataset)
    ))
