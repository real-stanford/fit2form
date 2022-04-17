import os
import ray
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from os.path import exists, splitext, getsize
import h5py
from torch import tensor, stack, cat
from numpy import load, array, savez
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from os import remove
from torch.nn import Module
from random import shuffle
from pathlib import Path
from enum import IntEnum
from glob import glob

from common_utils import tqdm_remote_get, glob_category


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input.view(input.size()[0], -1)
        return output


class ObjectDataset(Dataset):
    def __init__(self,
                 directory_path: str,
                 batch_size: int,
                 category_file: str = None,
                 check_validity=True):
        if category_file is not None and exists(category_file):
            print('[ObjectDataset] loading grasp objects from ',
                  directory_path, category_file)
            self.object_paths = glob_category(
                directory_path, category_file, "**/*graspobject.urdf")
        else:
            print('[ObjectDataset] globbing grasp objects ...')
            self.object_paths = [str(p)
                                 for p in Path(directory_path).rglob("*graspobject.urdf")]
        self.directory_path = directory_path
        total_count = len(self.object_paths)
        print(f'[ObjectDataset] found {total_count} objects')
        if check_validity:
            async_validity_filter = ray.remote(ObjectDataset.validity_filter)
            self.object_paths = [
                path for path in tqdm_remote_get(
                    task_handles=[
                        async_validity_filter.remote(object_path)
                        for object_path in self.object_paths
                    ],
                    desc="checking grasp objects ..."
                )
                if path is not None
            ]
            valid_count = len(self.object_paths)
            print('[ObjectDataset] found {} bad objects'.format(
                total_count - valid_count))
        self.batch_size = batch_size
        self.get_urdf_path_only = False
        self.loader = DataLoader(
            self,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=lambda batch: batch)
        self.loader_iter = iter(self.loader)
        self.iter_position = 0

    @staticmethod
    def check_validity(grasp_object_urdf_path: str, check_imprint=False):
        if not exists(grasp_object_urdf_path):
            return False
        prefix = splitext(grasp_object_urdf_path)[0]
        # make sure obj, collision mesh, and tsdf is there
        if not exists(prefix + '.obj'):
            return False
        elif not exists(prefix + '_collision.obj'):
            return False
        elif not exists(prefix + '_tsdf.npy'):
            return False
        elif check_imprint and (not exists(prefix + '_imprint.urdf')
                                or not exists(prefix + '_imprint_left_tsdf.npy')
                                or not exists(prefix + '_imprint_right_tsdf.npy')):
            return False
        return True

    @staticmethod
    def validity_filter(grasp_object_urdf_path: str):
        if not ObjectDataset.check_validity(grasp_object_urdf_path):
            return None
        return grasp_object_urdf_path

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, index):
        urdf_path = self.object_paths[index]
        if self.get_urdf_path_only:
            return str(urdf_path)
        prefix = splitext(urdf_path)[0]
        tsdf_path = prefix + '_tsdf.npy'
        tsdf = tensor(load(tsdf_path)).unsqueeze(dim=0)
        return {
            'tsdf': tsdf,
            'visual_mesh_path': prefix + '.obj',
            'collision_mesh_path': prefix + '_collision.obj',
            'tsdf_path': tsdf_path,
            'urdf_path': str(urdf_path)
        }

    def __iter__(self):
        self.iter_position = 0
        return self

    def __next__(self):
        if self.iter_position == len(self):
            raise StopIteration
        retval = self[self.iter_position]
        self.iter_position += 1
        return retval

    def sample(self):
        retval = next(self.loader_iter, None)
        if retval is None:
            self.loader_iter = iter(self.loader)
            retval = next(self.loader_iter, None)
        return retval


class ImprintObjectDataset(ObjectDataset):
    def __init__(self,
                 directory_path: str,
                 batch_size: int,
                 category_file: str,
                 check_validity=True):
        super(ImprintObjectDataset, self).__init__(
            directory_path, batch_size, category_file, check_validity=check_validity)
        total_count = len(self.object_paths)
        if check_validity:
            async_validity_filter = ray.remote(ImprintObjectDataset.validity_filter)
            self.object_paths = [
                path for path in tqdm_remote_get(
                    task_handles=[
                        async_validity_filter.remote(object_path)
                        for object_path in self.object_paths
                    ],
                    desc="checking grasp objects for presence of imprint fingers ..."
                )
                if path is not None
            ]
            valid_count = len(self.object_paths)
            print('[ImprintObjectDataset] found {} bad objects'.format(
                total_count - valid_count))

    @staticmethod
    def check_validity(grasp_object_urdf_path):
        is_valid = True
        prefix = splitext(grasp_object_urdf_path)[0]
        if not exists(prefix + "_imprint_left_collision.obj"):
            print("missing imprint left collision obj")
            is_valid = False
        elif not exists(prefix + "_imprint_left_tsdf.npy"):
            print("missing imprint left tsdf")
            is_valid = False
        elif not exists(prefix + "_imprint_right_collision.obj"):
            print("missing imprint right collision obj")
            is_valid = False
        elif not exists(prefix + "_imprint_right_tsdf.npy"):
            print("missing imprint right tsdf")
            is_valid = False
        elif not exists(prefix + "_imprint.urdf"):
            print("missing imprint urdf")
            is_valid = False
        return is_valid

    @staticmethod
    def validity_filter(grasp_object_urdf_path: str):
        if not ImprintObjectDataset.check_validity(grasp_object_urdf_path):
            return None
        return grasp_object_urdf_path


def get_loader(dataset: Dataset,
               batch_size: int = 32,
               collate_fn=None,
               distributed=False,
               shuffle=True,
               drop_last=True):
    kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'drop_last': drop_last,
        'shuffle': shuffle 
    }
    if distributed:
        kwargs['sampler'] = DistributedSampler(dataset)
        kwargs['shuffle'] = False
    if collate_fn:
        kwargs['collate_fn'] = collate_fn
    return DataLoader(dataset, **kwargs)

class GraspDatasetType(IntEnum):    # Enum for grasp dataset type
    PRETRAIN = 0
    COTRAIN = 1

class GraspDataset(Dataset):
    def __init__(self,
                 directory_path: str = None,
                 dataset_path: str = None,
                 batch_size: int = 32,
                 optimize_objectives=['grasp_success',
                                      'stability',
                                      'robustness'],
                 use_latest_points=None,
                 use_1_robustness=False,
                 grasp_dataset_type: GraspDatasetType = GraspDatasetType.PRETRAIN,
                 suffix: str = None,
                ):
        self.optimize_objectives = optimize_objectives
        self.use_1_robustness = use_1_robustness
        self.grasp_dataset_type = tensor(grasp_dataset_type).float()  # flag to identify the type of a particular datapoint
        self.design_objective_indices = get_design_objective_indices(
            self.optimize_objectives, self.use_1_robustness)
        self.directory_path = directory_path
        if directory_path is not None:
            self.hdf5_output_path = directory_path + (
                '/grasp_results.hdf5'
                if suffix is None else f'/grasp_results_{suffix}.hdf5')
            self.directory_path = directory_path
        elif dataset_path is not None:
            self.hdf5_output_path = dataset_path
        else:
            print("[GraspDataset] Supply directory_path or dataset_path!")
            exit(-1)
        self.batch_size = batch_size
        self.keys = []
        self.success_indices = []
        self.failure_indices = []
        self.use_latest_points = use_latest_points
        if exists(self.hdf5_output_path):
            with h5py.File(self.hdf5_output_path, "r") as f:
                self.keys.extend([k for k in f])
                print(f'[GraspDataset] Already has {len(self.keys)} points')
        else:
            with h5py.File(self.hdf5_output_path, "w") as f:
                pass
        hdf_path = Path(self.hdf5_output_path)
        hdf_dir = hdf_path.parents[0]
        hdf_stem = hdf_path.stem
        self.ind_cache_path = hdf_dir.joinpath(hdf_stem + "_indices.npz")
        if Path.exists(self.ind_cache_path):
            npzfile = load(self.ind_cache_path)
            self.success_indices = list(npzfile["success_indices"])
            self.failure_indices = list(npzfile["failure_indices"])
            print(f"[GraspDataset]loading indices from cache {self.ind_cache_path}",
                  f'{len(self.success_indices)} success and {len(self.failure_indices)} failure cases.')

    def extend(self, grasp_results, log=True):
        newly_added_indices = list()
        with h5py.File(self.hdf5_output_path, "a") as f:
            for grasp_result in grasp_results:
                group_key = '{:07d}'.format(len(self.keys))
                group = f.create_group(group_key)
                self.keys.append(group_key)
                newly_added_indices.append(len(self.keys) - 1)
                success = grasp_result['grasp_result'][0]
                if success:
                    self.success_indices.append(len(self.keys) - 1)
                else:
                    self.failure_indices.append(len(self.keys) - 1)
                dataset = group.create_dataset(
                    name='grasp_result',
                    data=grasp_result['grasp_result']
                )
                dataset.attrs["grasp_object_tsdf_path"] = \
                    grasp_result['grasp_object_tsdf_path']
                if 'left_finger_tsdf' in grasp_result\
                        and 'right_finger_tsdf' in grasp_result:
                    group.create_dataset(
                        name='left_finger_tsdf',
                        data=grasp_result['left_finger_tsdf'],
                        compression='gzip',
                        compression_opts=6
                    )
                    group.create_dataset(
                        name='right_finger_tsdf',
                        data=grasp_result['right_finger_tsdf'],
                        compression='gzip',
                        compression_opts=6
                    )
                elif 'left_finger_tsdf_path' in grasp_result\
                        and 'right_finger_tsdf_path' in grasp_result:
                    group.create_dataset(
                        name='left_finger_tsdf_path',
                        data=grasp_result['left_finger_tsdf_path']
                    )
                    group.create_dataset(
                        name='right_finger_tsdf_path',
                        data=grasp_result['right_finger_tsdf_path']
                    )
                else:
                    raise Exception(
                        "[Grasp Dataset] Not enough grasp information.")
            assert len(self.keys) == len(f)
        if log:
            print("\t[GraspDataset] count: {} , size: {}".format(
                len(self),
                getsize(self.hdf5_output_path)))
        return newly_added_indices

    def __len__(self):
        if not self.use_latest_points:
            return len(self.keys)
        else:
            return min(len(self.keys), self.use_latest_points)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_output_path, 'r') as f:
            if self.use_latest_points:
                index = -self.use_latest_points + index
            group = f.get(self.keys[index])
            grasp_metrics = group['grasp_result']
            grasp_object_tsdf_path = str(
                grasp_metrics.attrs['grasp_object_tsdf_path'])
            if splitext(grasp_object_tsdf_path)[1] == '.urdf':
                grasp_object_tsdf_path = splitext(grasp_object_tsdf_path)[
                    0] + '_tsdf.npy'
            grasp_object_tsdf = load(grasp_object_tsdf_path, allow_pickle=True)
            grasp_object_tsdf = tensor(grasp_object_tsdf).unsqueeze(dim=0)
            if 'left_finger_tsdf' in group and 'right_finger_tsdf' in group:
                left_finger_tsdf = tensor(group['left_finger_tsdf'])
                right_finger_tsdf = tensor(group['right_finger_tsdf'])
            elif 'left_finger_tsdf_path' in group \
                    and 'right_finger_tsdf_path' in group:
                left_finger_tsdf_path = str(
                    array(group['left_finger_tsdf_path']))
                right_finger_tsdf_path = str(
                    array(group['right_finger_tsdf_path']))
                left_finger_tsdf = tensor(
                    load(left_finger_tsdf_path))
                right_finger_tsdf = tensor(
                    load(right_finger_tsdf_path))

            left_finger_tsdf = left_finger_tsdf.squeeze().unsqueeze(dim=0)
            right_finger_tsdf = right_finger_tsdf.squeeze().unsqueeze(dim=0)
            grasp_metrics = tensor(grasp_metrics)[self.design_objective_indices]

            return (
                grasp_object_tsdf.float(),
                left_finger_tsdf.float(),
                right_finger_tsdf.float(),
                grasp_metrics.float(),
                self.grasp_dataset_type,
            )

    def get_indices_distribution(self):
        if self.use_latest_points:
            success_indices = [idx for idx in self.success_indices
                               if idx > len(self.keys) - self.use_latest_points]
            failure_indices = [idx for idx in self.failure_indices
                               if idx > len(self.keys) - self.use_latest_points]
        else:
            if len(self.success_indices) + len(self.failure_indices) != len(self):
                self.success_indices = list()
                self.failure_indices = list()
                for i, item in tqdm(
                        enumerate(self),
                        dynamic_ncols=True,
                        desc='[GraspDataset] analyzing grasp dataset',
                        smoothing=0.05,
                        total=len(self)):
                    _, _, _, grasp_metrics, _ = item
                    if grasp_metrics[0] == 1.0:
                        self.success_indices.append(i)
                    else:
                        self.failure_indices.append(i)
                savez(self.ind_cache_path,
                    success_indices=self.success_indices,
                    failure_indices=self.failure_indices)
            success_indices = self.success_indices
            failure_indices = self.failure_indices
        return success_indices, failure_indices

    def get_average_success_rate(self):
        success_indices, failure_indices = self.get_indices_distribution()
        if len(success_indices) + len(failure_indices) == 0:
            return 0
        return float(len(success_indices)) / (len(success_indices) + len(failure_indices))


class ConcatGraspDataset(GraspDataset):
    def __init__(self, grasp_datasets):
        self.dataset = ConcatDataset(grasp_datasets)
        self.success_indices = list()
        self.failure_indices = list()
        self.use_latest_points = None
        offset = 0
        for grasp_dataset in grasp_datasets:
            success_indices, failure_indices = grasp_dataset.get_indices_distribution()
            self.success_indices.extend(array(success_indices) + offset)
            self.failure_indices.extend(array(failure_indices) + offset)
            offset += len(success_indices) + len(failure_indices)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_indices_distribution(self):
        return self.success_indices, self.failure_indices

class BalancedGraspDataset(GraspDataset):
    # A grasp dataset which returns samples
    # based on target success ratio of 50%
    def __init__(self,
                 directory_path: str = None,
                 dataset_path: str = None,
                 batch_size: int = 32,
                 optimize_objectives=[],
                 grasp_dataset_type: GraspDatasetType=GraspDatasetType.PRETRAIN):
        super().__init__(directory_path=directory_path,
                         dataset_path=dataset_path,
                         batch_size=batch_size,
                         optimize_objectives=optimize_objectives,
                         grasp_dataset_type=grasp_dataset_type)
        self.update_indices()

    def update_indices(self):
        hdf_path = Path(self.hdf5_output_path)
        hdf_dir = hdf_path.parents[0]
        hdf_stem = hdf_path.stem
        ind_cache_path = hdf_dir.joinpath(hdf_stem + "_indices.npz")
        if Path.exists(ind_cache_path):
            npzfile = load(ind_cache_path)
            self.success_indices = npzfile["success_indices"]
            self.failure_indices = npzfile["failure_indices"]
            with h5py.File(self.hdf5_output_path, 'r') as f:
                if len(self.success_indices) + len(self.failure_indices) == len(f):
                    print(
                        f"[BalancedGraspDataset]loading indices from cache {ind_cache_path}")
                    print(
                        f"[BalancedGraspDataset] Found {len(self.success_indices)}",
                        f" success cases and {len(self.failure_indices)} failure cases"
                    )
                    return
        self.success_indices = []
        self.failure_indices = []
        with h5py.File(self.hdf5_output_path, 'r') as f:
            for i, key in tqdm(
                    enumerate(self.keys),
                    dynamic_ncols=True,
                    desc='[BalancedGraspDataset] analyzing grasp dataset',
                    smoothing=0.05,
                    total=len(f)):
                group = f.get(key)
                grasp_metrics = array(group['grasp_result'])
                if grasp_metrics[0] == 1.0:
                    self.success_indices.append(i)
                else:
                    self.failure_indices.append(i)
        savez(ind_cache_path, success_indices=self.success_indices,
              failure_indices=self.failure_indices)
        print(
            f"[BalancedGraspDataset] cached indices at location {ind_cache_path}")
        print(
            f"[BalancedGraspDataset] Found {len(self.success_indices)}",
            f" success cases and {len(self.failure_indices)} failure cases"
        )

    def extend(self, grasp_results, log=True):
        raise NotImplementedError

    def shuffle(self):
        print("[BalancedGraspDataset] shuffled indices for ",
              self.hdf5_output_path)
        shuffle(self.success_indices)
        shuffle(self.failure_indices)

    def __len__(self):
        return min(len(self.success_indices),
                   len(self.failure_indices))

    def __getitem__(self, index):
        failure_case = super().__getitem__(self.failure_indices[index])
        success_case = super().__getitem__(self.success_indices[index])
        grasp_object_tsdf = stack([failure_case[0], success_case[0]])
        left_finger_tsdf = stack([failure_case[1], success_case[1]])
        right_finger_tsdf = stack([failure_case[2], success_case[2]])
        grasp_metrics = stack([failure_case[3], success_case[3]])
        return (
            grasp_object_tsdf,
            left_finger_tsdf,
            right_finger_tsdf,
            grasp_metrics,
            self.grasp_dataset_type,
        )


def grasp_dataset_concat_collate_fn(batch):
    grasp_object_tsdfs = []
    left_finger_tsdfs = []
    right_finger_tsdfs = []
    grasp_metrics = []
    grasp_dataset_types = []
    for grasp_object_tsdf,\
            left_finger_tsdf,\
            right_finger_tsdf,\
            grasp_metric,\
            grasp_dataset_type in batch:
        grasp_object_tsdfs.append(
            grasp_object_tsdf
            if len(grasp_object_tsdf.size()) == 5
            else grasp_object_tsdf.unsqueeze(dim=1))
        left_finger_tsdf = left_finger_tsdf\
            if len(left_finger_tsdf.size()) == 5\
            else left_finger_tsdf.unsqueeze(dim=1)
        left_finger_tsdfs.append(left_finger_tsdf)
        right_finger_tsdfs.append(
            right_finger_tsdf
            if len(right_finger_tsdf.size()) == 5
            else right_finger_tsdf.unsqueeze(dim=1))
        grasp_metrics.append(grasp_metric)
        grasp_dataset_types.append(grasp_dataset_type)
    grasp_object_tsdfs = cat(tuple(grasp_object_tsdfs))
    left_finger_tsdfs = cat(tuple(left_finger_tsdfs))
    right_finger_tsdfs = cat(tuple(right_finger_tsdfs))
    grasp_metrics = cat(tuple(grasp_metrics))
    grasp_dataset_types = cat(tuple(grasp_dataset_types))

    return grasp_object_tsdfs, left_finger_tsdfs,\
        right_finger_tsdfs, grasp_metrics, grasp_dataset_types


class VAEDataset(Dataset):
    def __init__(self, finger_tsdf_paths, batch_size):
        self.batch_size = batch_size
        self.finger_tsdf_paths = finger_tsdf_paths

    def __len__(self):
        return len(self.finger_tsdf_paths)

    def __getitem__(self, idx):
        return tensor(load(self.finger_tsdf_paths[idx])).unsqueeze(dim=0).float()

    def get_loader(self):
        return DataLoader(self,
                          batch_size=self.batch_size,
                          num_workers=6,
                          shuffle=True)


class VAEDatasetHDF(Dataset):
    """
    VAE dataset using a graspdataset.hdf file
    """

    def __init__(self, dataset_path):
        assert Path(dataset_path).exists()
        self.dataset_path = dataset_path
        self.keys = []
        with h5py.File(self.dataset_path, "r") as f:
            self.keys.extend([k for k in f])
            print(f'[VAEDatasetHDF] Found 2 * {len(self.keys)} points')

    def __len__(self):
        return 2 * len(self.keys)

    def __getitem__(self, idx):
        hdf_index = idx // 2
        finger_tsdf_key = "left_finger_tsdf" if idx % 2 == 0 \
            else "right_finger_tsdf"
        finger_tsdf_path_key = "left_finger_tsdf_path" if idx % 2 == 0 \
            else "right_finger_tsdf_path"
        with h5py.File(self.dataset_path, 'r') as f:
            group = f.get(self.keys[hdf_index])
            assert finger_tsdf_key in group or finger_tsdf_path_key in group
            if finger_tsdf_key in group:
                finger_tsdf = tensor(
                    array(group[finger_tsdf_key])).squeeze().unsqueeze(dim=0).float()
            elif finger_tsdf_path_key in group:
                finger_tsdf_path = str(array(group[finger_tsdf_path_key]))
                finger_tsdf = tensor(load(finger_tsdf_path)).unsqueeze(dim=0).float()
            else:
                assert False, f"Neither {finger_tsdf_key}, nor {finger_tsdf_path_key} found in {self.dataset_path}[{idx}]"
            return finger_tsdf


def get_combined_loader(*loaders):
    for batches in zip(*loaders):
        yield grasp_dataset_concat_collate_fn(batches)


def get_balanced_cotrain_loader(
        cotrain_dataset: GraspDataset,
        pretrain_dataset: GraspDataset,
        batch_size: int):
    """
    Assuming batch contains 50-50 split of cotrain and pretrain,
    returns a loader for batches which balances success rate 
    amongst both cotrain and pretrain dataset while using entirety
    of cotrain by compensating cotrain failure with pretrain success
    and vice versa.
    """
    cotrain_loader = get_loader(
        dataset=cotrain_dataset,
        batch_size=int(batch_size / 2))
    cotrain_success_rate = cotrain_dataset.get_average_success_rate()
    target_pretrain_success_rate = 1 - cotrain_success_rate

    pretrain_dataset.get_average_success_rate()
    success_pretrain_dataset = Subset(
        pretrain_dataset, pretrain_dataset.success_indices)
    failure_pretrain_dataset = Subset(
        pretrain_dataset, pretrain_dataset.failure_indices)

    success_pretrain_loader = get_loader(
        dataset=success_pretrain_dataset,
        batch_size=int((batch_size / 2) * target_pretrain_success_rate))
    failure_pretrain_loader = get_loader(
        dataset=failure_pretrain_dataset,
        batch_size=int((batch_size / 2) * cotrain_success_rate))

    return get_combined_loader(
        cotrain_loader,
        success_pretrain_loader,
        failure_pretrain_loader
    )

def get_design_objective_indices(optimize_objectives, use_1_robustness=None):
    design_objective_indices = []
    if 'success' in optimize_objectives:
        design_objective_indices.extend(range(0, 1))
    if 'stability' in optimize_objectives:
        design_objective_indices.extend(range(1, 5))
    if 'robustness' in optimize_objectives:
        design_objective_indices.extend(range(5, 11))
    if use_1_robustness and \
        'success' in optimize_objectives and \
        'stability' in optimize_objectives and \
        'robustness' not in optimize_objectives:
        design_objective_indices.extend(range(5, 6))
    return design_objective_indices 