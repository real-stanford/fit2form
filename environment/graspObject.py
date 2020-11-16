import numpy as np
from os.path import splitext, exists
from .meshUtils import (
    Mesh,
    create_collision_mesh,
    create_grasp_object_urdf
)
import pybullet as p
from .utils import wait_until_object_is_stable
from .pybulletObject import PyBulletObject
from os import remove
import quaternion


class GraspObject(PyBulletObject):
    def __init__(self, urdf_path):
        super().__init__(urdf_path)

    def get_bounding_box(self):
        bb_min, bb_max = p.getAABB(self.id)
        bb_min = np.array(bb_min)
        bb_max = np.array(bb_max)
        # sometimes the bounding box is below plane, which causes trouble.
        # see https://github.com/bulletphysics/bullet3/issues/2820
        bb_min[2] = 0
        return bb_min, bb_max

    def rescale_to_bounds(self, output_prefix, view_bounds,
                          step_simulation_fn=None,
                          create_new_collision_mesh=True,
                          stability_orientation_tolerance=1e-2):

        # load mesh and collision mesh
        grasp_object_prefix = splitext(self.urdf_path)[0]
        mesh = Mesh(path=grasp_object_prefix
                    + '.obj')

        _, orientation = self.get_pose()

        mesh.rotate(*orientation,)
        scale = mesh.scale_to_bounds(view_bounds)
        translation = mesh.to_ground()
        mesh_output_path = output_prefix + '.obj'
        if not mesh.save(mesh_output_path):
            if exists(mesh_output_path):
                remove(mesh_output_path)
            return False

        collision_mesh_output_path = output_prefix + '_collision.obj'

        if create_new_collision_mesh:
            collision_mesh_output_path = create_collision_mesh(
                mesh_path=mesh_output_path)
        else:
            collision_mesh = Mesh(path=grasp_object_prefix
                                  + '_collision.obj')
            collision_mesh.rotate(*orientation)
            collision_mesh.scale(scale)
            collision_mesh.translate(translation)
            if not collision_mesh.save(collision_mesh_output_path):
                if exists(mesh_output_path):
                    remove(mesh_output_path)
                if mesh_output_path(collision_mesh_output_path):
                    remove(collision_mesh_output_path)
                return False

        urdf_output_path = create_grasp_object_urdf(
            mesh_path=mesh_output_path,
            collision_mesh_path=collision_mesh_output_path)

        # Load new URDF
        p.removeBody(self.id)
        try:
            self.id = p.loadURDF(urdf_output_path)
        except Exception as e:
            print(e)
            print(urdf_output_path)
            return False
        self.urdf_path = urdf_output_path

        loaded_orientation = np.quaternion(*self.get_pose()[1])

        if not wait_until_object_is_stable(
            grasp_object=self,
            max_wait_steps=10000,
                step_simulation_fn=step_simulation_fn):
            remove(mesh_output_path)
            remove(collision_mesh_output_path)
            remove(urdf_output_path)
            return False
        stable_orientation = np.quaternion(*self.get_pose()[1])
        orientation_displacement =\
            (stable_orientation * loaded_orientation.inverse()).angle()
        if orientation_displacement > stability_orientation_tolerance:
            # Object wobbled too much, which means collision mesh
            # did a bad job approximating original mesh
            return False
        return True
