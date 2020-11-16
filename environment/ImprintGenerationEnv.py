from .simCam import SimCam
from numpy import save
from .baseEnv import BaseEnv
from .graspObject import GraspObject
from .utils import wait_until_object_is_stable
from .meshUtils import (
    dump_gripper_urdf,
    create_collision_mesh,
    get_single_biggest_cc_single
)
from os.path import splitext, exists, basename
from .WSG50 import WSG50
from scipy.ndimage import rotate, shift
from .tsdfHelper import TSDFHelper


class ImprintGenerationEnv(BaseEnv):
    def __init__(self, config: dict, gui: bool):
        super().__init__(config, gui)
        distance = 15
        fov = 5
        self.right_cam = SimCam(
            position=[distance, 0, 0.3],
            lookat=[0, 0, 0.3],
            up_direction=[0, 0, 1],
            image_size=(512, 512),
            z_near=distance - 5.0,
            z_far=distance + 5.0,
            fov_w=fov
        )
        self.left_cam = SimCam(
            position=[-distance, 0, 0.3],
            lookat=[0, 0, 0.3],
            up_direction=[0, 0, 1],
            image_size=(512, 512),
            z_near=distance - 5.0,
            z_far=distance + 5.0,
            fov_w=fov
        )
        self.set_plane_visibility(False)

    def create_imprint_gripper_fingers(self,
                                       grasp_object_urdf_path: str,
                                       skip_if_already_exist=True):
        output_prefix = splitext(grasp_object_urdf_path)[0] + '_imprint'
        if skip_if_already_exist and exists(output_prefix + '.urdf'):
            return output_prefix + '.urdf'
        with GraspObject(urdf_path=grasp_object_urdf_path) as grasp_object:
            if not wait_until_object_is_stable(
                    grasp_object=grasp_object,
                    max_wait_steps=10000,
                    step_simulation_fn=self.step_simulation):
                self.error('Grasp Object not stable', grasp_object_urdf_path)
                return None

            # generate left finger
            left_finger_visual_mesh_path,\
                left_finger_collision_mesh_path =\
                self.create_imprint_gripper_finger(
                    self.left_cam,
                    output_prefix,
                    postfix='left',
                    rotation=(0, 180, 0))
            if (left_finger_visual_mesh_path is None or
                    not exists(left_finger_visual_mesh_path)) or\
                    (left_finger_collision_mesh_path is None or
                     not exists(left_finger_collision_mesh_path)):
                return None

            # generate right gripper
            right_finger_visual_mesh_path,\
                right_finger_collision_mesh_path = \
                self.create_imprint_gripper_finger(
                    self.right_cam,
                    output_prefix,
                    postfix='right',
                    rotation=(180, 180, 0))
            if (right_finger_visual_mesh_path is None or
                    not exists(right_finger_visual_mesh_path)) or\
                    (right_finger_collision_mesh_path is None or
                     not exists(right_finger_collision_mesh_path)):
                return None

            # combine left and right finger to gripper
            dump_gripper_urdf(
                basename(left_finger_collision_mesh_path),
                basename(left_finger_collision_mesh_path),
                basename(right_finger_collision_mesh_path),
                basename(right_finger_collision_mesh_path),
                output_prefix + '.urdf'
            )

            # visualize gripper
            if self.gui:
                self.set_plane_visibility(True)
                with WSG50(urdf_path=output_prefix + '.urdf',
                           max_grasp_force=self.max_grasp_force,
                           step_simulation_fn=self.step_simulation) as gripper:
                    print('Imprint gripper ' +
                          ('is' if gripper.is_valid() else 'is not')
                          + ' valid')
                    gripper.open()
                    gripper.move_to_floor()
                    gripper.close()
                    gripper.move_to_rest_height()
                    self.set_plane_visibility(False)

            return output_prefix + '.urdf'

    def create_imprint_gripper_finger(
            self,
            camera,
            output_prefix,
            postfix,
            rotation,
            close_off_mesh=True):
        # create tsdf
        color_im, depth_im = camera.get_image()
        finger_tsdf = TSDFHelper.tsdf_from_camera_data(
            views=[(*camera.get_image(),
                    camera.intrinsic_matrix,
                    camera.pose_matrix)],
            bounds=self.view_bounds,
            voxel_size=self.tsdf_voxel_size)

        # flip tsdf
        finger_tsdf = - finger_tsdf

        # rotate tsdf
        for angle, axes in zip(rotation, [(1, 0), (2, 1), (2, 0)]):
            if angle == 0:
                continue
            finger_tsdf = rotate(finger_tsdf,
                                 angle=angle,
                                 axes=axes)

        # shift closer to center
        size = finger_tsdf.shape[0]
        finger_tsdf = get_single_biggest_cc_single(
            shift(input=finger_tsdf,
                  shift=(int(size / 2), 0, 0),
                  cval=-1),
            return_num_components=False)
        save(output_prefix + f'_{postfix}_tsdf.npy',
             finger_tsdf)

        # create mesh from tsdf
        mesh_output_path = output_prefix + f'_{postfix}.obj'
        TSDFHelper.to_mesh(
            tsdf=finger_tsdf,
            voxel_size=self.tsdf_voxel_size,
            path=mesh_output_path)

        # create collision mesh from mesh
        collision_mesh_path = create_collision_mesh(
            mesh_path=mesh_output_path,
            high_res=True)
        return mesh_output_path, collision_mesh_path
