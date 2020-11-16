from .pybulletObject import PyBulletObject
import pybullet as p
from numpy import ones


class WSG50(PyBulletObject):

    def __init__(self,
                 urdf_path,
                 max_grasp_force=3,
                 max_grasp_speed=3,
                 step_simulation_fn=None):
        super().__init__(urdf_path)
        self.rest_height = 3
        self.speed = 0.01
        self.max_force = max_grasp_force
        self.step_simulation_fn = step_simulation_fn
        if self.step_simulation_fn is None:
            self.step_simulation_fn = p.stepSimulation

    def on_urdf_loaded(self):
        self.joint_ids = [
            p.getJointInfo(self.id, joint_idx)
            for joint_idx in range(p.getNumJoints(self.id))
        ]
        self.joint_ids = [
            joint_info[0]
            for joint_info in self.joint_ids
            if joint_info[2] != p.JOINT_FIXED
        ]
        self.set_pose(
            [0, 0, self.rest_height],
            [0, 0, 0, 1]
        )
        # set up constraints
        self.height_constraint = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            # jointType=p.JOINT_PRISMATIC, TODO get prismatic joint working
            # jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.set_height(self.rest_height)

    def is_valid(self):
        right_finger = self.get_link_idx(name='finger_right')
        right_mount = self.get_link_idx(name='mount_right')
        left_finger = self.get_link_idx(name='finger_left')
        left_mount = self.get_link_idx(name='mount_left')
        right_finger_connected_to_mount = \
            len(p.getClosestPoints(
                bodyA=self.id,
                bodyB=self.id,
                linkIndexA=right_finger,
                linkIndexB=right_mount,
                distance=0)) > 0
        left_finger_connected_to_mount = \
            len(p.getClosestPoints(
                bodyA=self.id,
                bodyB=self.id,
                linkIndexA=left_finger,
                linkIndexB=left_mount,
                distance=0)) > 0
        return right_finger_connected_to_mount\
            and left_finger_connected_to_mount

    def move_to_rest_height(self):
        return self.move_to_height(self.rest_height)

    def move_to_floor(self,
                      ground_plane_id=0,
                      increment_size=0.01,
                      max_steps=10000):
        """
        Lower height of gripper to just above the floor
        """
        steps = 0
        while not self.check_collision(
                other_id=ground_plane_id):
            self.set_height(self.get_height() - increment_size)
            self.step_simulation_fn()
            steps += 1
            if steps > max_steps:
                return False
        while self.check_collision(
                other_id=ground_plane_id):
            self.set_height(self.get_height() + increment_size)
            for _ in range(10):
                self.step_simulation_fn()
            steps += 1
            if steps > max_steps:
                return False
        return True

    def open(self, speed=None):
        self.control_gripper_joints([0, 0])

    def close(self, speed=None):
        self.control_gripper_joints([0.55, -0.55])

    def get_height(self):
        return self.get_pose()[0][2]

    def set_height(self, height):
        p.changeConstraint(
            userConstraintUniqueId=self.height_constraint,
            jointChildPivot=[0, 0, height],
            maxForce=40)

    def move_to_height(self,
                       height,
                       step_size=0.02,
                       max_steps=10000):
        steps = 0
        current_height = self.get_height()
        move_up = (height - current_height) > 0
        if not move_up:
            step_size = - step_size

        def done(curr, target):
            return curr > target\
                if move_up else\
                curr < target
        while not done(current_height, height):
            current_height += step_size
            self.set_height(current_height)
            self.step_simulation_fn()
            current_height = self.get_height()
            steps += 1
            if steps > max_steps:
                return False
        return True

    def control_gripper_joints(self, positions, speed=None):
        if speed is None:
            speed = self.speed
        p.setJointMotorControlArray(
            self.id,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=positions,
            forces=[
                self.max_force,
                self.max_force
            ],
            positionGains=speed * ones(2)
        )
        for _ in range(500):
            self.step_simulation_fn()
