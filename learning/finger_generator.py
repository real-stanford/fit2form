import abc
import copy
import torch
from learning.gripperDesigner import GripperDesigner
from utils import dicts_get
from torch.cuda import is_available as is_cuda_available
from torch import stack, no_grad, cat, rot90, tensor
from torch.optim import Adam
from copy import deepcopy
import os
from environment.tsdfHelper import TSDFHelper
from scipy.ndimage import rotate
from os.path import splitext
from numpy import load


class FingerGenerator(abc.ABC):
    def __init__(self, finger_generator_config):
        self.device = 'cuda' if is_cuda_available() else 'cpu'
        self.finetune_steps = finger_generator_config.get("finetune_steps", 0)
        self.lr = finger_generator_config.get("lr", 0)
        self.optim_grasp_success_only = \
            finger_generator_config.get('optimize_grasp_success_only', False)
        self.desc = finger_generator_config.get("desc", "")
        self.robustness_angles = finger_generator_config[
            'train_config']['environment']['robustness_angles']
        # append 0 to also optimize for default orientation
        self.robustness_angles.append(0)
        self.robustness_angles = sorted(self.robustness_angles)
        self.base_path = f"/tmp/finetuned_grippers/{self.desc}"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    @abc.abstractmethod
    def create_fingers(self, grasp_object_tsdfs):
        pass


class ImprintFingerGenerator(FingerGenerator):
    def __init__(self, finger_generator_config):
        super(ImprintFingerGenerator, self).__init__()
        assert finger_generator_config["type"] == "imprint_gn"

    def create_fingers(self, grasp_objects):
        left_fingers = [splitext(grasp_object['urdf_path'])[0] + '_imprint_left_tsdf.npy'
                        for grasp_object in grasp_objects]
        right_fingers = [splitext(grasp_object['urdf_path'])[0] + '_imprint_right_tsdf.npy'
                         for grasp_object in grasp_objects]
        urdf_paths = [splitext(grasp_object['urdf_path'])[0] + '_imprint.urdf'
                      for grasp_object in grasp_objects]
        left_fingers = [tensor(load(left_finger))
                        for left_finger in left_fingers]
        right_fingers = [tensor(load(right_finger))
                         for right_finger in right_fingers]
        left_fingers = stack(left_fingers).unsqueeze(dim=1)
        right_fingers = stack(right_fingers).unsqueeze(dim=1)
        return left_fingers, right_fingers, urdf_paths


class TrainedGNFingerGenerator(FingerGenerator):
    def __init__(self, finger_generator_config):
        super(TrainedGNFingerGenerator, self).__init__(finger_generator_config)
        assert finger_generator_config["type"] == "trained_gn"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        checkpoint_dict = {
            "fn_checkpoint_path": finger_generator_config["checkpoint_dict"]["fn_checkpoint_path"]\
                if "fn_checkpoint_path" in finger_generator_config["checkpoint_dict"] else None,
            "gn_checkpoint_path": finger_generator_config["checkpoint_dict"]["gn_checkpoint_path"]\
                if "gn_checkpoint_path" in finger_generator_config["checkpoint_dict"] else None,
        }

        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"]
            if "fitness_class" in hyperparameters else None,
            generator_class_str=hyperparameters["generator_class"]
            if "generator_class" in hyperparameters else None,
            checkpoint_dict=checkpoint_dict,
            load_fitness_net=False
        )
        self.net.eval()

    def create_fingers(self, grasp_objects):
        grasp_object_tsdfs = stack(
            dicts_get(grasp_objects, 'tsdf')).to(self.device)
        with no_grad():
            left_fingers, right_fingers = self.net.create_fingers(
                grasp_object_tsdfs)
        left_fingers = left_fingers.cpu()
        right_fingers = right_fingers.cpu()
        return left_fingers, right_fingers


class FinetuneGNFingerGenerator(FingerGenerator):
    def __init__(self, finger_generator_config):
        super(FinetuneGNFingerGenerator, self).__init__(
            finger_generator_config)
        assert finger_generator_config["type"] == "finetune_gn"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"]
            if "fitness_class" in hyperparameters else None,
            checkpoint_dict=finger_generator_config["checkpoint_dict"],
            load_fitness_net=True,
        )
        self.gn_net_orig = deepcopy(self.net.generator_net)
        self.finetune_steps = finger_generator_config["finetune_steps"]
        self.lr = finger_generator_config["lr"]
        self.optim_grasp_success_only = False
        if "optimize_grasp_success_only" in finger_generator_config\
                and finger_generator_config["optimize_grasp_success_only"]:
            self.optim_grasp_success_only = True
        self.desc = finger_generator_config["desc"]

    def create_fingers(self, grasp_objects):
        left_fingers, right_fingers = torch.empty(
            (0, 1, 40, 40, 40)), torch.empty((0, 1, 40, 40, 40))
        grasp_object_tsdfs = dicts_get(grasp_objects, 'tsdf')

        for grasp_object_index, grasp_object_tsdf in enumerate(grasp_object_tsdfs):
            self.net.eval()
            go_batch = grasp_object_tsdf.unsqueeze(0).to(self.device)
            gn_optim = Adam(self.net.generator_net.parameters(), lr=1e-5)
            for i in range(self.finetune_steps + 1):
                gn_optim.zero_grad()
                lf_batch, rf_batch = self.net.create_fingers(go_batch)
                if i != self.finetune_steps:
                    fitness_pred = self.net.evaluate_fingers(
                        go_batch, lf_batch, rf_batch)
                    generator_loss = -fitness_pred
                    if self.optim_grasp_success_only:
                        generator_loss[0, 0].backward()
                    else:
                        generator_loss.mean().backward()
                    gn_optim.step()
                else:
                    left_fingers = cat(
                        (left_fingers, lf_batch.detach().cpu()), dim=0)
                    right_fingers = cat(
                        (right_fingers, rf_batch.detach().cpu()), dim=0)
            self.net.generator_net = deepcopy(self.gn_net_orig)
        return left_fingers, right_fingers


class FinetuneFingerGenerator(FingerGenerator):
    """
    Finetune generated fingers
    """

    def __init__(self, finger_generator_config):
        super(FinetuneFingerGenerator, self).__init__(finger_generator_config)
        assert finger_generator_config["type"] == "finetune_gn_fingers"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"] if "fitness_class" in hyperparameters else None,
            checkpoint_dict=finger_generator_config["checkpoint_dict"],
            load_fitness_net=True,
        )
        self.gn_net_orig = deepcopy(self.net.generator_net)
        self.finetune_steps = finger_generator_config["finetune_steps"]
        self.lr = finger_generator_config["lr"]
        self.optim_grasp_success_only = False
        if "optimize_grasp_success_only" in finger_generator_config\
                and finger_generator_config["optimize_grasp_success_only"]:
            self.optim_grasp_success_only = True
        self.desc = finger_generator_config["desc"]

    def create_fingers(self, grasp_objects):
        left_fingers, right_fingers = torch.empty(
            (0, 1, 40, 40, 40)), torch.empty((0, 1, 40, 40, 40))
        grasp_object_tsdfs = dicts_get(grasp_objects, 'tsdf')

        for grasp_object_index, grasp_object_tsdf in enumerate(grasp_object_tsdfs):
            self.net.eval()
            go_batch = grasp_object_tsdf.unsqueeze(0).to(self.device)
            with no_grad():
                lf_batch, rf_batch = self.net.create_fingers(go_batch)
            lf_batch.requires_grad_()
            rf_batch.requires_grad_()
            finger_optim = Adam([lf_batch, rf_batch], lr=1e-5)
            for i in range(self.finetune_steps + 1):
                finger_optim.zero_grad()
                if i != self.finetune_steps:
                    fitness_pred = self.net.evaluate_fingers(
                        go_batch, lf_batch, rf_batch)
                    generator_loss = -fitness_pred
                    if self.optim_grasp_success_only:
                        generator_loss[0].backward()
                    else:
                        generator_loss.mean().backward()
                    finger_optim.step()
                else:
                    left_fingers = cat(
                        (left_fingers, lf_batch.detach().cpu()), dim=0)
                    right_fingers = cat(
                        (right_fingers, rf_batch.detach().cpu()), dim=0)
                # dump_grasp_tuple(lf_batch.detach().clone(), go_batch.detach().clone(), rf_batch.detach().clone(), os.path.join(self.base_path, f"{grasp_object_index}_step_{i}.obj"))
            self.net.generator_net = deepcopy(self.gn_net_orig)
        return left_fingers, right_fingers


class RobustnessFinetuneFingersFingerGenerator(FingerGenerator):
    """
    Finetune generated fingers for robustness by *optimizing generated fingers* to be good for all object orientations
    """

    def __init__(self, finger_generator_config):
        super(RobustnessFinetuneFingersFingerGenerator,
              self).__init__(finger_generator_config)
        assert finger_generator_config["type"] == "finetune_gn_fingers_robustness"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"] if "fitness_class" in hyperparameters else None,
            checkpoint_dict=finger_generator_config["checkpoint_dict"],
            load_fitness_net=True,
        )

    def create_fingers(self, grasp_objects):
        left_fingers, right_fingers = torch.empty(
            (0, 1, 40, 40, 40)), torch.empty((0, 1, 40, 40, 40))
        grasp_object_tsdfs = dicts_get(grasp_objects, 'tsdf')

        for grasp_object_index, grasp_object_tsdf in enumerate(grasp_object_tsdfs):
            # Create starting fingers using generator network
            self.net.eval()
            go_batch = grasp_object_tsdf.unsqueeze(0).to(self.device)
            with torch.no_grad():
                lf_batch, rf_batch = self.net.create_fingers(go_batch)
            lf_batch.requires_grad_()
            rf_batch.requires_grad_()
            finger_optimizer = Adam([lf_batch, rf_batch], lr=self.lr)
            # optimize fingers for all objects
            for i in range(self.finetune_steps):
                finger_optimizer.zero_grad()
                # collect loss one by one
                sum_robustness_loss = None
                for robustness_angle in self.robustness_angles:
                    # TODO: verify that the tsdf out has value 1
                    rotated_object = torch.tensor(
                        rotate(go_batch.cpu().numpy(), angle=robustness_angle, axes=(
                            2, 3), reshape=False, mode="constant", cval=1),  # rotate in x-y plane
                        device=self.device
                    )
                    fitness_pred = self.net.evaluate_fingers(
                        rotated_object, lf_batch, rf_batch)
                    if sum_robustness_loss is None:
                        sum_robustness_loss = -fitness_pred
                    else:
                        sum_robustness_loss += -fitness_pred
                robustness_loss = sum_robustness_loss / \
                    len(self.robustness_angles)
                if self.optim_grasp_success_only:
                    robustness_loss[0].backward()
                else:
                    robustness_loss.mean().backward()
                finger_optimizer.step()
            left_fingers = cat((left_fingers, lf_batch.detach().cpu()), dim=0)
            right_fingers = cat(
                (right_fingers, rf_batch.detach().cpu()), dim=0)
        return left_fingers, right_fingers


class RobustnessFinetuneGNFingerGenerator(FingerGenerator):
    """
    Finetune *generator network* such that the generated fingers are robust
        by optimizing parameters of generator network
    """

    def __init__(self, finger_generator_config):
        super(RobustnessFinetuneGNFingerGenerator,
              self).__init__(finger_generator_config)
        assert finger_generator_config["type"] == "finetune_gn_robustness"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"] if "fitness_class" in hyperparameters else None,
            checkpoint_dict=finger_generator_config["checkpoint_dict"],
            load_fitness_net=True,
        )

    def create_fingers(self, grasp_objects):
        left_fingers, right_fingers = torch.empty(
            (0, 1, 40, 40, 40)), torch.empty((0, 1, 40, 40, 40))
        grasp_object_tsdfs = dicts_get(grasp_objects, 'tsdf')

        for grasp_object_index, grasp_object_tsdf in enumerate(grasp_object_tsdfs):
            # Create starting fingers using generator network
            self.net.eval()
            go_batch = grasp_object_tsdf.unsqueeze(0).to(self.device)
            gn_optimizer = Adam(
                self.net.generator_net.parameters(), lr=self.lr)
            # optimize fingers for all objects
            for i in range(self.finetune_steps):
                gn_optimizer.zero_grad()
                lf_batch, rf_batch = self.net.create_fingers(go_batch)
                # collect loss one by one
                sum_robustness_loss = None
                for robustness_angle in self.robustness_angles:
                    rotated_object = torch.tensor(
                        rotate(go_batch.cpu().numpy(), angle=robustness_angle, axes=(
                            2, 3), reshape=False, mode="constant", cval=1),  # rotate in x-y plane
                        device=self.device
                    )
                    fitness_pred = self.net.evaluate_fingers(
                        rotated_object, lf_batch, rf_batch)
                    if sum_robustness_loss is None:
                        sum_robustness_loss = -fitness_pred
                    else:
                        sum_robustness_loss += -fitness_pred
                robustness_loss = sum_robustness_loss / \
                    len(self.robustness_angles)
                if self.optim_grasp_success_only:
                    robustness_loss[0].backward()
                else:
                    robustness_loss.mean().backward()
                gn_optimizer.step()
            left_fingers = cat((left_fingers, lf_batch.detach().cpu()), dim=0)
            right_fingers = cat(
                (right_fingers, rf_batch.detach().cpu()), dim=0)
        return left_fingers, right_fingers


class FinetuneRandomFingerGenerator(FingerGenerator):
    def __init__(self, finger_generator_config):
        super(FinetuneRandomFingerGenerator, self).__init__(
            finger_generator_config)
        assert finger_generator_config["type"] == "finetune_random_fingers"
        # load network
        hyperparameters = finger_generator_config['train_config']['training']['hyperparameters']
        self.net = GripperDesigner(
            embedding_dim=hyperparameters['embedding_dim'],
            designer_weight_decay=hyperparameters['designer_weight_decay'],
            designer_lr=hyperparameters['designer_lr'],
            vae_weight_decay=hyperparameters['vae_weight_decay'],
            vae_lr=hyperparameters['vae_lr'],
            device=self.device,
            fitness_class_str=hyperparameters["fitness_class"] if "fitness_class" in hyperparameters else None,
            checkpoint_dict=finger_generator_config["checkpoint_dict"],
            load_fitness_net=True,
        )

    def create_fingers(self, grasp_objects):
        left_fingers, right_fingers = torch.empty(
            (0, 1, 40, 40, 40)), torch.empty((0, 1, 40, 40, 40))
        grasp_object_tsdfs = dicts_get(grasp_objects, 'tsdf')
        self.net.eval()
        go_batch = grasp_object_tsdf.to(self.device)
        lf_emb_batch = 2 * \
            torch.rand((len(grasp_object_tsdfs),
                        self.net.embedding_dim), device=self.device) - 1
        rf_emb_batch = 2 * \
            torch.rand((len(grasp_object_tsdfs),
                        self.net.embedding_dim), device=self.device) - 1
        lf_emb_batch.requires_grad_()
        rf_emb_batch.requires_grad_()

        finger_emb_opt = Adam([lf_emb_batch, rf_emb_batch], lr=self.lr)
        for i in range(self.finetune_steps + 1):
            finger_emb_opt.zero_grad()
            if i != self.finetune_steps:
                lf_batch = self.net.decoder(lf_emb_batch)
                rf_batch = self.net.decoder(rf_emb_batch)
                loss = -self.net.evaluate_fingers(go_batch, lf_batch, rf_batch)
                if self.optim_grasp_success_only:
                    loss[0].backward()
                else:
                    loss.mean().backward()
                finger_emb_opt.step()
            else:
                left_fingers = cat(
                    (left_fingers, lf_batch.detach().cpu()), dim=0)
                right_fingers = cat(
                    (right_fingers, rf_batch.detach().cpu()), dim=0)
        return left_fingers, right_fingers


def get_finger_generator(finger_generator_config):
    if finger_generator_config["type"] == "trained_gn":
        return TrainedGNFingerGenerator(finger_generator_config)
    elif finger_generator_config["type"] == "finetune_gn":
        return FinetuneGNFingerGenerator(finger_generator_config)
    elif finger_generator_config["type"] == "finetune_gn_fingers":
        return FinetuneFingerGenerator(finger_generator_config)
    elif finger_generator_config["type"] == "finetune_gn_fingers_robustness":
        return RobustnessFinetuneFingersFingerGenerator(finger_generator_config)
    elif finger_generator_config["type"] == "finetune_gn_robustness":
        return RobustnessFinetuneGNFingerGenerator(finger_generator_config)
    elif finger_generator_config["type"] == "finetune_random_fingers":
        return FinetuneRandomFingerGenerator(finger_generator_config)
    else:
        raise NotImplementedError


def dump_grasp_tuple(left_finger, grasp_object, right_finger, path):
    left_finger = rot90(left_finger, 2, (3, 4))
    right_finger = rot90(right_finger, 2, (2, 4))
    input_tsdf = cat([left_finger, grasp_object, right_finger], dim=2)
    TSDFHelper.to_mesh(tsdf=input_tsdf[0, 0, :].cpu(
    ).numpy(), voxel_size=0.015, path=path)
