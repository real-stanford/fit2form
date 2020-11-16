import pybullet as p
from .utils import wait_until_object_is_stable
from .WSG50 import WSG50
from .baseEnv import BaseEnv
from .graspObject import GraspObject
from .meshUtils import create_gripper, check_base_connection
from time import sleep
from os.path import splitext, dirname, basename
from numpy import array, mean, pi
from .MeshRenderer import MeshRenderer
from .simCam import revolving_shot
from .recorder import PybulletRecorder


class GraspSimulationEnv(BaseEnv):

    def __init__(self, config: dict, gui: bool = False):
        super().__init__(config, gui)
        env_conf = config['environment']
        self.grasp_metrics = env_conf['grasp_metrics']
        self.stability_config = env_conf['stability']
        self.stability_force_directions = \
            self.stability_config['force_directions']
        self.stability_force_magnitude = \
            self.stability_config['force_magnitude']
        self.stability_force_steps =  \
            self.stability_config['steps_to_apply_force']
        self.stability_forces = [
            array(force_direction) * self.stability_force_magnitude
            for force_direction in self.stability_force_directions
        ]
        self.robustness_angles = env_conf['robustness_angles']
        self.average_robustness = env_conf['average_robustness']
        self.measurement_fns = [("success", self.check_grasp)]
        if 'stability' in self.grasp_metrics:
            self.measurement_fns.extend([
                (
                    f'stability_{idx + 1}',
                    lambda obj_id, gripper, force=stability_force:
                    self.check_force_stability(
                        obj_id,
                        gripper,
                        force)
                )
                for idx, stability_force in enumerate(
                    self.stability_forces)
            ])

        if 'robustness' in self.grasp_metrics:
            if self.average_robustness:
                self.measurement_fns.append((
                    'robustness',
                    self.check_all_robustness))
            else:
                self.measurement_fns.extend([
                    (
                        f'robustness_{idx}',
                        lambda obj_id, gripper, angle=robustness_angle:
                        self.check_robustness(
                            obj_id,
                            gripper,
                            angle)
                    )
                    for idx, robustness_angle in enumerate(
                        self.robustness_angles)
                ])
        self.measurement_fns = dict(self.measurement_fns)
        # self.renderer = MeshRenderer()
        self.renderer = None

    def compute_finger_grasp_score(self,
                                   left_finger_tsdf,
                                   right_finger_tsdf,
                                   grasp_object_urdf_path,
                                   urdf_output_path_prefix,
                                   visualize=False):
        if not check_base_connection(right_finger_tsdf)\
                or not check_base_connection(left_finger_tsdf):
            return {
                'score': self.failure_score(),
                'base_connected': False,
                'created_grippers_failed': None,
                'single_connected_component': None,
                'grasp_object_path': grasp_object_urdf_path
            }
        # 1. create gripper meshes
        retval = create_gripper(
            left_finger=left_finger_tsdf.squeeze(),
            right_finger=right_finger_tsdf.squeeze(),
            urdf_output_path_prefix=urdf_output_path_prefix,
            voxel_size=self.tsdf_voxel_size)
        if retval is None:
            return {
                'score': self.failure_score(),
                'base_connected': True,
                'created_grippers_failed': True,
                'single_connected_component': None,
                'grasp_object_path': grasp_object_urdf_path
            }
        gripper_urdf_path, single_connected_component = retval

        # 2. simulate the grippers
        return {
            'score': self.simulate_grasp(
                grasp_object_urdf_path=grasp_object_urdf_path,
                gripper_urdf_path=gripper_urdf_path,
                left_finger_tsdf=left_finger_tsdf,
                right_finger_tsdf=right_finger_tsdf,
                visualize=visualize),
            'base_connected': True,
            'created_grippers_failed': False,
            'single_connected_component': single_connected_component,
            'grasp_object_path': grasp_object_urdf_path
        }

    def failure_score(self):
        return dict([(metric, 0.0)
                     for metric in self.measurement_fns])

    def simulate_grasp(self,
                       grasp_object_urdf_path,
                       gripper_urdf_path,
                       left_finger_tsdf=None,
                       right_finger_tsdf=None,
                       visualize=False):
        significantly_base_connected = True
        if left_finger_tsdf is not None:
            significantly_base_connected =\
                check_base_connection(left_finger_tsdf)
        if right_finger_tsdf is not None:
            significantly_base_connected = significantly_base_connected \
                and check_base_connection(right_finger_tsdf)

        if not significantly_base_connected:
            return self.failure_score()

        self.visualize = False
        if visualize:
            visualize_prefix = splitext(gripper_urdf_path)[0]
            visualize_prefix = visualize_prefix[:-8]
            simulation_pickle_file_prefix = visualize_prefix
            # simulation_pickle_file_prefix = dirname(grasp_object_urdf_path)
            # simulation_pickle_file_prefix = dirname(visualize_prefix) \
            # + "/" + basename(simulation_pickle_file_prefix)
            self.renderer.render_mesh(
                mesh_path=visualize_prefix + '_left_collision.obj',
                color=MeshRenderer.GREEN,
                upside_down=True)
            self.renderer.render_mesh(
                mesh_path=visualize_prefix + '_right_collision.obj',
                color=MeshRenderer.RED,
                upside_down=True)
            grasp_object_prefix = splitext(grasp_object_urdf_path)[0]
            self.renderer.render_mesh(
                mesh_path=grasp_object_prefix + '.obj',
                output_prefix=visualize_prefix + '_graspobject',
                color=MeshRenderer.GREY,
                z_height=0.3)
            self.step_count = 0
            self.visualization_images = []
            self.visualization_text = \
                f'Executing grasp with {self.max_grasp_force}N'
        with GraspObject(urdf_path=grasp_object_urdf_path) as grasp_object,\
                WSG50(urdf_path=gripper_urdf_path,
                      max_grasp_force=self.max_grasp_force,
                      max_grasp_speed=self.max_grasp_speed,
                      step_simulation_fn=self.step_simulation) as gripper:
            if not gripper.is_valid():
                self.error('Invalid Gripper', gripper_urdf_path)
                if self.gui:
                    sleep(5)
                if self.visualize:
                    self.save_results(
                        self.failure_score(),
                        visualize_prefix + '_results.txt')
                return self.failure_score()

            if not wait_until_object_is_stable(
                    grasp_object=grasp_object,
                    max_wait_steps=10000,
                    step_simulation_fn=self.step_simulation):
                self.error('Grasp Object not stable', grasp_object_urdf_path)
                return None

            # Gripper is valid and object is stable
            # Save state for robustness test later on
            self.stable_state = p.saveState()

            # Perform grasp
            gripper.open()
            self.visualize = visualize
            if visualize:
                self.recorder = PybulletRecorder()
                self.recorder.register_object(
                    body_id=gripper.id,
                    path=gripper_urdf_path
                )
                self.recorder.register_object(
                    body_id=grasp_object.id,
                    path=grasp_object_urdf_path
                )
            if not gripper.move_to_floor():
                self.error("Gripper can't move to floor",
                           gripper_urdf_path)
                return None
            gripper.close()
            if visualize and self.record_gifs:
                self.save_recording(
                    path=visualize_prefix + '_grasp.gif',
                    images=revolving_shot([0, 0, 0.3]))
            if not gripper.move_to_rest_height():
                self.error("Gripper can't move to rest height",
                           gripper_urdf_path)
                return None
            if visualize and self.record_gifs:
                self.save_recording(
                    path=visualize_prefix + '_postgrasp.gif',
                    images=revolving_shot([0, 0, 1.7]))

            # If grasp unsuccessful then return all zero scores
            if not self.measurement_fns['success'](
                    grasp_object, gripper):
                if self.visualize:
                    self.save_recording(
                        path=visualize_prefix + '.gif')
                    self.save_results(
                        self.failure_score(),
                        visualize_prefix + '_results.txt')
                    self.recorder.save(
                        path=simulation_pickle_file_prefix + '.pkl')
                    self.recorder.reset()
                return self.failure_score()

            # Save state for other grasp metrics tests
            self.successful_grasp_state = p.saveState()

            # Otherwise, measure other grasp metrics
            grasp_score = [('success', 1.0)]
            for metric in self.measurement_fns:
                if metric == 'success':
                    continue
                self.visualization_text = 'Measuring ' + metric
                if self.gui:
                    print(f'\t{self.visualization_text}')
                grasp_score.append((
                    metric,
                    float(self.measurement_fns[metric](grasp_object, gripper))
                ))
            grasp_score = dict(grasp_score)
            if self.visualize:
                self.save_recording(path=visualize_prefix + '.gif')
                self.save_results(
                    grasp_score,
                    visualize_prefix + '_results.txt')
                self.recorder.save(
                    path=simulation_pickle_file_prefix + '.pkl')
                self.recorder.reset()
            return grasp_score

    def save_results(self, score_dict, path):
        with open(path, 'w') as f:
            f.write(''.join([str(int(score))
                             for score in score_dict.values()]))

    def check_grasp(self, grasp_object, gripper):
        for _ in range(200):
            self.step_simulation()
        retval = gripper.check_collision(grasp_object.id)
        if self.gui:
            print('\t\t', 'success' if retval else 'fail')
        return retval

    def check_force_stability(self, grasp_object, gripper, force):
        self.visualization_text += f' at {self.stability_force_magnitude}N'
        p.restoreState(self.successful_grasp_state)
        for i in range(self.stability_force_steps):
            p.applyExternalForce(grasp_object.id,
                                 -1,
                                 force,
                                 [0, 0, 0],
                                 p.WORLD_FRAME)
            self.step_simulation()
        for _ in range(200):
            self.step_simulation()
        return self.check_grasp(grasp_object, gripper)

    def check_all_robustness(self, grasp_object, gripper):
        results = []
        for angle in self.robustness_angles:
            results.append(self.check_robustness(grasp_object, gripper, angle))
        return mean(results)

    def check_robustness(self, grasp_object, gripper, angle):
        self.visualization_text = f'robustness at {angle} degrees'
        p.restoreState(self.stable_state)
        # rotate object by angle
        grasp_object.rotate(0, 0, pi * angle / 180)

        wait_until_object_is_stable(
            grasp_object=grasp_object,
            max_wait_steps=10000,
            step_simulation_fn=self.step_simulation)

        # attempt grasp
        gripper.open()
        if not gripper.move_to_floor():
            return 0.
        gripper.close()
        if not gripper.move_to_rest_height():
            return 0.
        retval = float(self.check_grasp(grasp_object, gripper))
        p.restoreState(self.successful_grasp_state)
        return retval

    @staticmethod
    def error(message, detail):
        print(f'[ERROR] {message}: {detail}')
