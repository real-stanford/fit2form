from .utils import Flatten, get_design_objective_indices
from torch.nn import (
    Module,
    Sequential,
    Linear,
    LeakyReLU,
    Sigmoid,
    Conv3d,
    ReLU,
    BatchNorm1d,
    BatchNorm3d
)
from torch import cat
from torch import rot90
import os
from environment.tsdfHelper import TSDFHelper


class BasicFitnessNet(Module):
    def __init__(self, optimize_objectives, use_1_robustness = None):
        super(BasicFitnessNet, self).__init__()
        self.optimize_objectives = optimize_objectives
        self.use_1_robustness = use_1_robustness
        self.output_dim = 0
        design_objective_indices = get_design_objective_indices(optimize_objectives, use_1_robustness)
        self.output_dim = len(design_objective_indices)
        self.net = self.get_net()

    def forward(self, input):
        return self.net(input)

    def get_net(self):
        return Sequential(
            Conv3d(3, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Conv3d(8, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 64, kernel_size=3, stride=2),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Flatten(),
            Linear(5832, 2048),
            LeakyReLU(),
            Linear(2048, 2048),
            BatchNorm1d(2048),
            LeakyReLU(),
            Linear(2048, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            Linear(1024, 512),
            BatchNorm1d(512),
            LeakyReLU(),
            Linear(512, 256),
            BatchNorm1d(256),
            LeakyReLU(),
            Linear(256, 128),
            BatchNorm1d(128),
            LeakyReLU(),
            Linear(128, self.output_dim),
            Sigmoid()
        )


class BasicBlock(Module):
    def __init__(self, inplanes, planes,
                 kernel_size, stride,
                 padding=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.net = Sequential(
            Conv3d(inplanes,
                   planes,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   bias=False),
            BatchNorm3d(planes),
            LeakyReLU()
        )

    def forward(self, input):
        return self.net(input)


class ResidualBlock(Module):
    def __init__(self, inplanes, planes,
                 kernel_size, stride,
                 norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm3d
        self.planes = planes
        self.stride = stride
        self.conv1 = Conv3d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv3d(planes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResFitnessNet(BasicFitnessNet):
    def __init__(self, optimize_objectives):
        super(ResFitnessNet, self).__init__(optimize_objectives)

    def get_net(self):
        return Sequential(
            BasicBlock(3, 32, 3, 2),
            #
            ResidualBlock(32, 32, 3, 1),
            BasicBlock(32, 64, 3, 2),
            #
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            BasicBlock(64, 128, 3, 2),
            #
            ResidualBlock(128, 128, 3, 1),
            ResidualBlock(128, 128, 3, 1),
            BasicBlock(128, 256, 3, 2),
            #
            ResidualBlock(256, 256, 3, 1),
            ResidualBlock(256, 256, 3, 1),
            BasicBlock(256, 512, 3, 2),
            #
            ResidualBlock(512, 512, 3, 1),
            ResidualBlock(512, 512, 3, 1),
            BasicBlock(512, 1024, 3, 2),
            #
            ResidualBlock(1024, 1024, 3, 1),
            ResidualBlock(1024, 1024, 3, 1),
            BasicBlock(1024, 1024, 3, 2),
            #
            Flatten(),
            Linear(1024, 512),
            BatchNorm1d(512),
            LeakyReLU(),
            Linear(512, 256),
            BatchNorm1d(256),
            LeakyReLU(),
            Linear(256, 128),
            BatchNorm1d(128),
            LeakyReLU(),
            Linear(128, self.output_dim),
            Sigmoid()
        )


class SpatialConcatResFitnessNet(BasicFitnessNet):
    """
    Idea:
    (1) the low-level geometry is more important than highlevel geometry,
        so should have more weight before downsampling
    (2) remove part of finger volume that doesn't matter for grasping

    Input volume dimension:
    [40] [40] [40] concat along spatial dimension to get [120 x 40 x 40]
    """

    def __init__(self, optimize_objectives, use_1_robustness = None):
        super(SpatialConcatResFitnessNet, self).__init__(optimize_objectives, use_1_robustness)

    def dump_meshes(self, suffix, inp):
        # Dump these volumes
        dump_path = "/tmp/meshes"
        print(f"Dumping meshes to {dump_path}")
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        for i in range(inp.shape[0]):
            TSDFHelper.to_mesh(tsdf=inp[i, 0, :].cpu().numpy(
            ), voxel_size=0.015, path=os.path.join(dump_path, f"{i}_{suffix}.obj"))

    def forward(self, input):
        assert len(input.shape) == 5
        grasp_object = input[:, 0, :, :, :].unsqueeze(dim=1)
        left_finger = input[:, 1, :, :, :].unsqueeze(dim=1)
        right_finger = input[:, 2, :, :, :].unsqueeze(dim=1)

        left_finger = rot90(left_finger, 2, (3, 4))
        right_finger = rot90(right_finger, 2, (2, 4))
        input_tsdf = cat([left_finger, grasp_object, right_finger], dim=2)

        return self.net(input_tsdf)

    def get_net(self):
        return Sequential(
            BasicBlock(1, 32,
                       kernel_size=5,
                       stride=2,
                       padding=2),
            #
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
            BasicBlock(32, 64, 3, 2),
            #
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            BasicBlock(64, 128, 3, 2),
            #
            ResidualBlock(128, 128, 3, 1),
            BasicBlock(128, 256, 3, 2),
            #
            ResidualBlock(256, 256, 3, 1),
            BasicBlock(256, 512, 3, 2),
            #
            ResidualBlock(512, 512, 3, 1),
            BasicBlock(512, 1024, 3, 2),
            #
            ResidualBlock(1024, 1024, 3, 1),
            BasicBlock(1024, 1024, 3, 2),
            #
            Flatten(),
            Linear(1024, 512),
            BatchNorm1d(512),
            LeakyReLU(),
            Linear(512, 256),
            BatchNorm1d(256),
            LeakyReLU(),
            Linear(256, 128),
            BatchNorm1d(128),
            LeakyReLU(),
            Linear(128, self.output_dim),
            Sigmoid()
        )


class SpatialConcatResFitnessNetV2(SpatialConcatResFitnessNet):
    def __init__(self, optimize_objectives, use_1_robustness=None):
        super(SpatialConcatResFitnessNetV2, self).__init__(optimize_objectives, use_1_robustness)

    def get_net(self):
        return Sequential(
            BasicBlock(1, 32, 5, 2, 2),
            #
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
            BasicBlock(32, 64, 3, 2),
            #
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            BasicBlock(64, 128, 3, 2),
            #
            ResidualBlock(128, 128, 3, 1),
            ResidualBlock(128, 128, 3, 1),
            BasicBlock(128, 256, 3, 2),
            #
            ResidualBlock(256, 256, 3, 1),
            ResidualBlock(256, 256, 3, 1),
            BasicBlock(256, 512, 3, 2),
            #
            ResidualBlock(512, 512, 3, 1),
            ResidualBlock(512, 512, 3, 1),
            BasicBlock(512, 1024, 3, 2),
            #
            ResidualBlock(1024, 1024, 3, 1),
            ResidualBlock(1024, 1024, 3, 1),
            BasicBlock(1024, 1024, 3, 2),
            #
            Flatten(),
            Linear(1024, 512),
            BatchNorm1d(512),
            LeakyReLU(),
            Linear(512, 256),
            BatchNorm1d(256),
            LeakyReLU(),
            Linear(256, 128),
            BatchNorm1d(128),
            LeakyReLU(),
            Linear(128, self.output_dim),
            Sigmoid()
        )
