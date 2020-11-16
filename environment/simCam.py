import pybullet as p
from numpy import array, tan, pi, uint8, sin, cos
from math import atan
from numpy.linalg import inv


class SimCam(object):
    def __init__(self,
                 position,
                 lookat,
                 up_direction=[0, 0, 1],
                 image_size=(512, 512),
                 z_near=0.01,
                 z_far=20.0,
                 fov_w=70.0):
        self.image_size = image_size
        self.z_near = z_near
        self.z_far = z_far
        self.fov_w = fov_w
        self.focal_length = (float(self.image_size[1]) / 2)\
            / tan((pi * self.fov_w / 180) / 2)
        self.fov_h = (atan((float(self.image_size[0]) / 2)
                           / self.focal_length) * 2 / pi) * 180
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov_h,  # vertical FOV
            aspect=float(self.image_size[1]) / \
            float(self.image_size[0]),  # must be float
            nearVal=self.z_near,
            farVal=self.z_far
        )

        # compute intrinsic matrix
        self.intrinsic_matrix = array(
            [
                [self.focal_length, 0, float(self.image_size[1]) / 2],
                [0, self.focal_length, float(self.image_size[0]) / 2],
                [0, 0, 1]
            ]
        )

        self.set_pose(position, lookat, up_direction)

    def set_pose(self, position, lookat, up=[0, 0, 1]):
        self.pos = position
        self.lookat = lookat
        self.up_dir = up
        self.view_matrix = p.computeViewMatrix(
            self.pos, self.lookat, self.up_dir)
        self.pose_matrix = inv(
            array(self.view_matrix).reshape(4, 4).T)
        self.pose_matrix[:, 1:3] = -self.pose_matrix[:, 1:3]

    def get_image(self, shadows=False):
        img_arr = p.getCameraImage(
            self.image_size[0],
            self.image_size[1],
            self.view_matrix,
            self.proj_matrix,
            shadow=int(shadows),
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        w = img_arr[0]
        h = img_arr[1]
        rgb = img_arr[2]
        rgb_arr = array(rgb, dtype=uint8).reshape([h, w, 4])
        rgb = rgb_arr[:, :, 0:3]

        d = img_arr[3]
        d = array(d).reshape([h, w])
        d = (2.0 * self.z_near * self.z_far) \
            / (self.z_far + self.z_near - (2.0 * d - 1.0)
               * (self.z_far - self.z_near))
        return rgb, d


def revolving_shot(center=[0, 0, 0], count=60, radius=1.0):
    center = array(center)
    images = []
    camera = SimCam(position=center + array([0, 1, 0]), lookat=center)
    for i in range(count):
        position = center + array([
            sin(i * 2 * pi / count) * radius,
            cos(i * 2 * pi / count) * radius,
            0])
        camera.set_pose(position, center)
        images.append(camera.get_image(shadows=True)[0])
    return images
