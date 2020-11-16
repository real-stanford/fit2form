import pybullet as p
import pybullet_data
from time import sleep
from numpy import array
from .simCam import SimCam
from PIL import Image, ImageDraw, ImageFont
from imageio import get_writer
from pygifsicle import optimize


class BaseEnv:
    def __init__(self, config: dict, gui: bool,
                 use_gravity=True):
        self.gui = gui
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_body_id = p.loadURDF("plane.urdf")
        self.plane_visual_data = p.getVisualShapeData(self.plane_body_id)
        p.setRealTimeSimulation(0)
        if use_gravity:
            p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(enableFileCaching=0)

        env_conf = config['environment']
        self.tsdf_voxel_size = env_conf['tsdf_voxel_size']
        self.view_bounds = array(env_conf['view_bounds'])
        self.contact_normals = env_conf['contact_normals']
        # TODO add support contact normals
        if self.contact_normals:
            print("Contact Normal not currently supported")
            exit()
        self.max_grasp_force = env_conf['grasp']['max_grasp_force']
        self.max_grasp_speed = env_conf['grasp']['max_grasp_speed']

        # Visualization
        self.visualize = False
        self.step_count = 0
        self.visualization_cam = SimCam(
            position=[0, -2.5, 1.5],
            lookat=[0, 0, 1.5]
        )
        self.visualization_images = []

        # self.visualization_font = ImageFont.truetype(
        #     "/usr/share/fonts/truetype/lato/Lato-Black.ttf", 32)
        self.visualization_font = None
        self.record_gifs = False
        self.recorder = None

    def step_simulation(self):
        p.stepSimulation()
        if self.gui:
            sleep(1.0 / 240.)
        if self.recorder:
            self.recorder.add_keyframe()
        self.step_count += 1
        if self.visualize and self.record_gifs and self.step_count % 40 == 0:
            image = self.visualization_cam.get_image(shadows=True)[0]
            image = self.add_text_to_image(image, self.visualization_text)
            self.visualization_images.append(image)

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

    def initialize_recording(self):
        self.visualization_images = []
        self.step_count = 0

    def add_text_to_image(self, image, text, color='rgb(0, 0, 0)'):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        (x, y) = (0, 0)
        draw.text((x, y), text,
                  fill=color,
                  font=self.visualization_font)
        return array(image)

    def save_recording(self, path, images=None):
        if not self.record_gifs:
            return
        if not images:
            images = self.visualization_images
            clear_after_save = True
        else:
            clear_after_save = False
        if len(images) > 0:
            with get_writer(path, mode='I') as writer:
                for image in images:
                    writer.append_data(image)
            optimize(path)
            if clear_after_save:
                self.visualization_images = []
