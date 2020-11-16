import pybullet as p
from .simCam import SimCam
from .fusion import TSDFVolume
from numpy import sin, cos, pi, array
from skimage.measure import marching_cubes_lewiner
from numpy import copy


def export_obj(filename, verts, faces, norms, colors=None):
    # Write header
    obj_file = open(filename, 'w')

    # Write vertex list
    for i in range(verts.shape[0]):
        obj_file.write("v %f %f %f\n" %
                       (verts[i, 0], verts[i, 1], verts[i, 2]))

    for i in range(norms.shape[0]):
        obj_file.write("vn %f %f %f\n" %
                       (norms[i, 0], norms[i, 1], norms[i, 2]))

    faces = copy(faces)
    faces += 1

    for i in range(faces.shape[0]):
        obj_file.write("f %d %d %d\n" %
                       (faces[i, 0], faces[i, 1], faces[i, 2]))

    obj_file.close()


class TSDFHelper:
    def __init__(self,
                 view_bounds,
                 voxel_size,
                 lookat=[0, 0, 0],
                 safety_margin=0.2,
                 num_cameras_per_ring=10,
                 plane_id=0):
        p.setPhysicsEngineParameter(enableFileCaching=0)
        self.view_bounds = view_bounds
        self.voxel_size = voxel_size

        # setup cameras
        x_min, y_min, z_min = self.view_bounds[:, 0] - safety_margin
        x_max, y_max, z_max = self.view_bounds[:, 1] + safety_margin
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2

        params = [
            # Top face
            [[x_mid, y_mid, z_max], [0, 1, 0]],
            # Bottom face
            [[x_mid, y_mid, z_min], [0, 1, 0]],
        ]
        radius = max(y_max, x_max)
        for i in range(num_cameras_per_ring):
            x_pos = radius * sin(2 * pi * i / num_cameras_per_ring)
            y_pos = radius * cos(2 * pi * i / num_cameras_per_ring)
            params.append([[x_pos, y_pos, z_max], [0, 0, 1]])
            params.append([[x_pos, y_pos, z_mid], [0, 0, 1]])
            params.append([[x_pos, y_pos, z_min], [0, 0, 1]])

        self.cameras = [SimCam(
            position=param[0],
            lookat=lookat,
            up_direction=param[1],
        ) for param in params]

        self.plane_body_id = plane_id
        self.plane_visual_data = p.getVisualShapeData(self.plane_body_id)

    def set_plane_visibility(self, is_visible):
        if is_visible:
            p.changeVisualShape(
                self.plane_body_id,
                self.plane_visual_data[0][1],
                rgbaColor=self.plane_visual_data[0][7])
        else:
            p.changeVisualShape(
                self.plane_body_id,
                self.plane_visual_data[0][1],
                rgbaColor=[0, 0, 0, 0])

    def generate_scene_voxel(self):
        self.set_plane_visibility(False)
        voxel = TSDFHelper.tsdf_from_camera_data(
            views=[(
                *camera.get_image(),
                camera.intrinsic_matrix,
                camera.pose_matrix,
            )
                for camera in self.cameras],
            bounds=self.view_bounds,
            voxel_size=self.voxel_size)
        self.set_plane_visibility(True)
        return voxel

    @staticmethod
    def tsdf_from_camera_data(views, bounds, voxel_size):
        # Initialize voxel volume
        tsdf_vol = TSDFVolume(bounds, voxel_size=voxel_size)
        # Fuse different views to one voxel
        for view in views:
            tsdf_vol.integrate(*view, obs_weight=1.)
        return tsdf_vol._tsdf_vol_cpu

    @staticmethod
    def to_mesh(tsdf,
                path,
                voxel_size,
                vol_origin=[0, 0, 0],
                level=0.0):
        if type(tsdf) != array:
            tsdf = array(tsdf)
        # Block off sides to get valid marching cubes
        tsdf = copy(tsdf)
        tsdf[:, :, -1] = 1
        tsdf[:, :, 0] = 1
        tsdf[:, -1, :] = 1
        tsdf[:, 0, :] = 1
        tsdf[-1, :, :] = 1
        tsdf[0, :, :] = 1
        if tsdf.min() > level or tsdf.max() < level:
            return False
        try:
            verts, faces, norms, _ = marching_cubes_lewiner(
                tsdf,
                level=level)

            # Shift the origin from (bottom left back)
            # vertex to the center of the volume.
            verts = verts - array([*tsdf.shape]) / 2

            # scale to final volume size
            verts = verts * voxel_size

            # now move the mesh origin to world's vol_origin
            verts += vol_origin
            export_obj(path, verts, faces, norms)
            return True
        except Exception as e:
            print(e)
            return False
