from trimesh import load
from pyrender import Scene, PerspectiveCamera, PointLight, OffscreenRenderer, Mesh
from numpy import ones, pi, array, zeros
import pybullet as p
from matplotlib.pyplot import imsave
from os.path import splitext
from os import environ
if 'PYOPENGL_PLATFORM' not in environ or\
        environ['PYOPENGL_PLATFORM'] != 'egl':
    environ['PYOPENGL_PLATFORM'] = 'egl'


class MeshRenderer:
    RED = [225, 0, 0, 225]
    GREEN = [0, 255, 0, 225]
    BLUE = [0, 0, 255, 225]
    GREY = [100, 100, 100, 255]

    def __init__(self, image_size=(320, 320)):
        self.image_size = image_size
        self.scene = Scene(ambient_light=ones(3) * 0.2)
        camera = PerspectiveCamera(yfov=pi / 3.0, aspectRatio=1.0)
        self.scene.add(camera,
                       pose=array(p.computeViewMatrix(
                           cameraEyePosition=[0, 0, -1],
                           cameraTargetPosition=[0, 0, 0],
                           cameraUpVector=[0, 1, 0]
                       )).reshape((4, 4)).T)
        self.scene.add(PointLight(
            color=ones(3),
            intensity=10.0),
            pose=[
            [1, 0, 0, -2],
            [0, 1, 0, 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        self.scene.add(PointLight(
            color=ones(3),
            intensity=20.0),
            pose=[
            [1, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 1, -2],
            [0, 0, 0, 1]
        ])
        self.renderer = OffscreenRenderer(*self.image_size)

    def get_views(self, z_height, upside_down):
        up = -1 if upside_down else 1
        view1 = array(p.computeViewMatrix(
            cameraEyePosition=[0, -0.04 + z_height, 0.1],
            cameraTargetPosition=[2, 0.2, -1],
            cameraUpVector=[0, up, 0]
        )).reshape((4, 4)).T
        view2 = array(p.computeViewMatrix(
            cameraEyePosition=[0, -0.04 + z_height, -0.1],
            cameraTargetPosition=[2, 0.2, 1],
            cameraUpVector=[0, up, 0]
        )).reshape((4, 4)).T
        return view1, view2

    def render_mesh(self,
                    mesh_path,
                    output_prefix=None,
                    color=None,
                    z_height=0,
                    upside_down=False):
        if not output_prefix:
            output_prefix = splitext(mesh_path)[0]
        if not color:
            color = MeshRenderer.RED
        mesh = load(mesh_path)
        self.rotate_mesh(mesh, *p.getQuaternionFromEuler([-pi / 2, 0, 0]))
        self.color_mesh(mesh, color=color)
        self.center_mesh(mesh)
        mesh = Mesh.from_trimesh(mesh, smooth=False)

        view1, view2 = self.get_views(
            z_height=z_height,
            upside_down=upside_down)

        mesh_node = self.scene.add(mesh, pose=view1)
        color, depth = self.renderer.render(self.scene)
        with open(output_prefix + '_view_1_color.png', 'wb') as f:
            imsave(f, color)
        with open(output_prefix + '_view_1_depth.png', 'wb') as f:
            imsave(f, depth)
        self.scene.remove_node(mesh_node)

        mesh_node = self.scene.add(mesh, pose=view2)
        color, depth = self.renderer.render(self.scene)
        with open(output_prefix + '_view_2_color.png', 'wb') as f:
            imsave(f, color)
        with open(output_prefix + '_view_2_depth.png', 'wb') as f:
            imsave(f, depth)
        self.scene.remove_node(mesh_node)

    @staticmethod
    def center_mesh(mesh):
        mesh.vertices -= mesh.center_mass

    @staticmethod
    def color_mesh(mesh, color):
        mesh.visual.vertex_colors = color

    @staticmethod
    def rotate_mesh(mesh, x, y, z, w):
        matrix3x3 = p.getMatrixFromQuaternion([x, y, z, w])
        matrix4x4 = zeros((4, 4))
        matrix4x4[3, 3] = 0
        matrix4x4[0, 0:3] = matrix3x3[0:3]
        matrix4x4[1, 0:3] = matrix3x3[3:6]
        matrix4x4[2, 0:3] = matrix3x3[6:9]
        mesh.apply_transform(matrix4x4)
