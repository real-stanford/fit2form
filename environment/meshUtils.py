from subprocess import check_call
from os import devnull
from os.path import splitext, exists, basename
import trimesh
import pybullet as p
from numpy import array, zeros, mean, ones
from .urdfTemplates import grasp_object_template, wsg50_template
from skimage.morphology import label
import traceback
import sys
from .tsdfHelper import TSDFHelper
from scipy.ndimage.interpolation import zoom
from time import time


class Mesh:
    """
    A wrapper class around trimesh
    """

    def __init__(self, path):
        self.path = path
        try:
            self.tm_mesh = trimesh.load(
                path,
                process=False,
                split_object=True,
                force='scene'
            )
        except:
            pass

    def rotate(self, x, y, z, w):
        matrix3x3 = p.getMatrixFromQuaternion([x, y, z, w])
        matrix4x4 = zeros((4, 4))
        matrix4x4[3, 3] = 0
        matrix4x4[0, 0:3] = matrix3x3[0:3]
        matrix4x4[1, 0:3] = matrix3x3[3:6]
        matrix4x4[2, 0:3] = matrix3x3[6:9]
        self.tm_mesh.apply_transform(matrix4x4)

    def scale_to_bounds(self, view_bounds, scale=None):
        extent = self.tm_mesh.bounding_box.extents
        x_scale = (view_bounds[0][1] - view_bounds[0][0]) / extent[0]
        y_scale = (view_bounds[1][1] - view_bounds[1][0]) / extent[1]
        z_scale = (view_bounds[2][1] - view_bounds[2][0]) / extent[2]
        scale = (x_scale, y_scale, z_scale)
        self.tm_mesh.apply_scale(scale)
        return scale

    def scale(self, scale):
        self.tm_mesh.apply_scale(scale)

    def translate(self, translation):
        self.tm_mesh.apply_translation(translation)

    def to_ground(self):
        bounds = array(self.tm_mesh.bounds)
        x_mean = mean(bounds[:, 0])
        y_mean = mean(bounds[:, 1])
        z_min = bounds[0, 2]
        translation = [-x_mean, -y_mean, -z_min]
        self.tm_mesh.apply_translation(translation)
        return translation

    def save(self, path):
        try:
            self.tm_mesh.export(path,
                                include_texture=False,
                                include_normals=False,
                                include_color=False,
                                return_texture=False,
                                write_texture=False)
            return True
        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            return False


def create_collision_mesh(
        mesh_path: str,
        timeout: int = 3000,
        high_res=False, log=False):
    if not exists(mesh_path):
        return None
    output_path = splitext(mesh_path)[0] + '_collision.obj'
    with open(devnull, 'w') as FNULL:
        cmds = [
            "assets/vhacd",
            "--input", mesh_path,
            "--output", output_path
        ]
        if high_res:
            cmds.extend([
                "--resolution 10000000",
                "--depth 32",
                "--planeDownsampling 1",
                "--maxNumVerticesPerCH 1024",
                "--convexhullDownsampling 1",
                "--concavity 0.000001"
            ])
        else:
            cmds.append("--resolution 25000")
        try:
            start = time()
            check_call(
                cmds,
                stdout=FNULL,
                timeout=timeout)
            if log:
                print("{:.01f} seconds | {}".format(
                    float(time() - start),
                    mesh_path
                ))
            return output_path
        except Exception as e:
            print(e)
            return None


def create_grasp_object_urdf(
        mesh_path: str,
        collision_mesh_path=None):
    if collision_mesh_path is None:
        collision_mesh_path = splitext(
            mesh_path)[0] + '_collision.obj'
    if not exists(collision_mesh_path):
        return None
    urdf_output_path = splitext(mesh_path)[0] + '.urdf'
    collision_mesh_path = basename(collision_mesh_path)
    mesh_path = basename(mesh_path)
    urdf_content = grasp_object_template.format(
        collision_mesh_path,
        collision_mesh_path)
    with open(urdf_output_path, 'w') as f:
        f.write(urdf_content)
    return urdf_output_path


def check_base_connection(vol,
                          base_connection_width=4,
                          base_connection_threshold=0.1):
    vol = vol.squeeze()
    plane_area = vol.shape[0] * vol.shape[1]
    for plane_idx in range(base_connection_width):
        if (vol[:, :, plane_idx] <= 0).sum() >=\
                plane_area * base_connection_threshold:
            return True
    return False


def get_single_biggest_cc_single(tsdf_vol,
                                 return_num_components=True,
                                 level=0.0):
    # Create a mask which represents presence of volume
    binary_vol_mask = array(tsdf_vol < level, dtype=int)

    # label connected components for mask
    labeled_vol_mask, num_components = label(binary_vol_mask, return_num=True)
    max_size = 0
    max_size_component_i = 0  # 0 represents background

    for component_i in range(1, num_components + 1):
        size = (labeled_vol_mask == component_i).sum()
        if max_size < size:
            max_size = size
            max_size_component_i = component_i
    max_size_vol = array(
        labeled_vol_mask == max_size_component_i, dtype=int) * tsdf_vol + \
        ones(tsdf_vol.shape) * array(labeled_vol_mask != max_size_component_i)
    if return_num_components:
        return max_size_vol, num_components
    else:
        return max_size_vol


def dump_gripper_urdf(
        left_finger_visual_mesh,
        left_finger_collision_mesh,
        right_finger_visual_mesh,
        right_finger_collision_mesh,
        output_path,
        finger_z_offset=-1.35,
        mesh_scale=1.15,
        finger_x_offset=-0.2):
    urdf_content = wsg50_template.format(
        left_finger_visual_mesh=basename(left_finger_visual_mesh),
        left_finger_collision_mesh=basename(left_finger_collision_mesh),
        right_finger_visual_mesh=basename(right_finger_visual_mesh),
        right_finger_collision_mesh=basename(right_finger_collision_mesh),
        finger_z_offset=finger_z_offset,
        mesh_scale=mesh_scale,
        finger_x_offset=finger_x_offset,
    )
    with open(output_path, 'w') as f:
        f.write(urdf_content)
    return output_path


def prepare_finger(finger_tsdf, output_prefix, postfix, voxel_size, log=False, overwrite=False):
    mesh_path = f'{output_prefix}_{postfix}.obj'
    collision_mesh_path = f'{output_prefix}_{postfix}_collision.obj'
    finger_tsdf, num_components =\
        get_single_biggest_cc_single(finger_tsdf.numpy())
    if not overwrite and exists(collision_mesh_path):
        return collision_mesh_path, collision_mesh_path, num_components
    if not TSDFHelper.to_mesh(
            tsdf=finger_tsdf,
            voxel_size=voxel_size,
            path=mesh_path):
        if log:
            print('\t[MeshUtils] dump mesh from tsdf voxel failed: ' +
                  output_prefix)
        return None
    assert exists(mesh_path)
    collision_mesh_path = create_collision_mesh(mesh_path)
    if collision_mesh_path is None:
        if log:
            print(f'\t[MeshUtils] create {postfix} finger collision mesh failed:',
                  output_prefix)
        return None
    assert exists(collision_mesh_path)
    return mesh_path, collision_mesh_path, num_components


def create_gripper(
        left_finger,
        right_finger,
        urdf_output_path_prefix,
        voxel_size):
    rv = prepare_finger(
        left_finger, urdf_output_path_prefix, 'left', voxel_size)
    if not rv:
        return None
    left_finger_mesh_path, \
        left_finger_collision_mesh_path,\
        left_finger_num_components = rv
    rv = prepare_finger(
        right_finger, urdf_output_path_prefix, 'right', voxel_size)
    if not rv:
        return None
    right_finger_mesh_path, \
        right_finger_collision_mesh_path,\
        right_finger_num_components = rv

    urdf_output_path = urdf_output_path_prefix + '_gripper.urdf'

    dump_gripper_urdf(
        left_finger_visual_mesh=left_finger_collision_mesh_path,
        left_finger_collision_mesh=left_finger_collision_mesh_path,
        right_finger_visual_mesh=right_finger_collision_mesh_path,
        right_finger_collision_mesh=right_finger_collision_mesh_path,
        output_path=urdf_output_path
    )

    return urdf_output_path, int(
        left_finger_num_components == 1
        and right_finger_num_components == 1)


def chop_and_scale_finger(
        vol,
        voxel_size,
        output_mesh_path,
        base_connection_threshold=0.1):
    threshold_plane_found = False
    threshold_plane_index = None
    vol = vol.squeeze()
    assert len(vol.shape) == 3
    plane_area = vol.shape[0] * vol.shape[1]
    vol = get_single_biggest_cc_single(
        tsdf_vol=vol.numpy(), return_num_components=False)
    for plane_idx in range(0, (vol.shape[2]) // 3):
        threshold_plane_found = (vol[:, :, plane_idx] <= 0).sum() >= \
            plane_area * base_connection_threshold
        if threshold_plane_found:
            threshold_plane_index = plane_idx
            break

    # - if no suitable plane found. Repeat
    if not threshold_plane_found:
        return None

    # create and load the stretched version of chopped mesh in simulation VIA ZOOM method
    vol_chopped = vol[:, :, threshold_plane_index:]
    zoom_factors = [final_dim / initial_dim for final_dim, initial_dim in
                    zip(vol.shape, vol_chopped.shape)]
    vol_chopped_stretched = zoom(vol_chopped, zoom_factors)
    if not check_base_connection(vol_chopped_stretched):
        print("[ERROR] Not base connected after chop and scale:",
              output_mesh_path)
        return None
    if not TSDFHelper.to_mesh(
            tsdf=vol_chopped_stretched,
            voxel_size=voxel_size,
            path=output_mesh_path):
        return None

    collision_mesh_path = create_collision_mesh(output_mesh_path)
    if collision_mesh_path is None:
        return None
    return {
        'tsdf_volume': vol_chopped_stretched,
        'collision_mesh_path': collision_mesh_path,
        'visual_mesh_path': output_mesh_path
    }


def create_shapenet_gripper(
        left_finger_tsdf,
        right_finger_tsdf,
        output_prefix,
        voxel_size):
    chopped_left_finger = \
        chop_and_scale_finger(
            vol=left_finger_tsdf,
            voxel_size=voxel_size,
            output_mesh_path=output_prefix + '_left.obj'
        )
    if not chopped_left_finger:
        return None
    chopped_right_finger = \
        chop_and_scale_finger(
            vol=right_finger_tsdf,
            voxel_size=voxel_size,
            output_mesh_path=output_prefix + '_right.obj'
        )
    if not chopped_right_finger:
        return None
    gripper_urdf_path = dump_gripper_urdf(
        left_finger_visual_mesh=basename(
            chopped_left_finger['visual_mesh_path']),
        left_finger_collision_mesh=basename(
            chopped_left_finger['collision_mesh_path']),
        right_finger_visual_mesh=basename(
            chopped_right_finger['visual_mesh_path']),
        right_finger_collision_mesh=basename(
            chopped_right_finger['collision_mesh_path']),
        output_path=output_prefix + '.urdf')
    return {
        'gripper_urdf_path': gripper_urdf_path,
        'left_finger_tsdf': chopped_left_finger['tsdf_volume'],
        'right_finger_tsdf': chopped_right_finger['tsdf_volume'],
    }
