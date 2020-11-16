from .baseEnv import BaseEnv
import pybullet as p
from random import random, seed
from numpy import pi, array, save
from .utils import wait_until_object_is_stable
from .tsdfHelper import TSDFHelper
from .graspObject import GraspObject
from os.path import splitext


class GraspObjectGenerationEnv(BaseEnv):
    def __init__(self, config: dict, gui: bool):
        super().__init__(config, gui)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        self.tsdf_helper = TSDFHelper(
            view_bounds=self.view_bounds,
            lookat=self.view_bounds.mean(axis=1),
            voxel_size=self.tsdf_voxel_size,
            plane_id=self.plane_body_id
        )
        seed()

    def create_grasp_object(self, urdf_path):
        # 1. load object
        with GraspObject(urdf_path=urdf_path) as grasp_object:
            euler_angles = array([random(), random(), random()])
            euler_angles = euler_angles * 4 * pi
            orientation = p.getQuaternionFromEuler(euler_angles)
            grasp_object.set_pose([0, 0, 1], orientation)
            # 2. let it settle
            if not wait_until_object_is_stable(
                    grasp_object=grasp_object,
                    max_wait_steps=10000,
                    step_simulation_fn=self.step_simulation):
                return None
            euler_angles = (euler_angles * 180 / pi).astype(int)
            output_prefix = splitext(urdf_path)[0] + \
                '_{:03d}_{:03d}_{:03d}_graspobject'.format(*euler_angles)

            # 3. rescale to view bounds
            if not grasp_object.rescale_to_bounds(
                    view_bounds=self.view_bounds,
                    output_prefix=output_prefix,
                    step_simulation_fn=self.step_simulation):
                return None
            # 4. Save TSDF
            grasp_object_tsdf = self.tsdf_helper.generate_scene_voxel()
            with open(output_prefix + '_tsdf.npy', 'wb') as f:
                save(f, grasp_object_tsdf)
            return output_prefix
