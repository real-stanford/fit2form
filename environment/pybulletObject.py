import pybullet as p
from numpy import array


class PyBulletObject:
    """
    Provides urdf as a resource which automatically
    cleans up after itself and other useful functions
    """

    def __init__(self, urdf_path):
        self.urdf_path = urdf_path

    def __enter__(self):
        try:
            self.id = p.loadURDF(
                self.urdf_path,
                flags=p.URDF_USE_SELF_COLLISION)
            self.on_urdf_loaded()
            return self
        except Exception as e:
            print(e)
            print(self.urdf_path)
            exit()

    def on_urdf_loaded(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        p.removeBody(self.id)

    def set_pose(self, position, orientation):
        p.resetBasePositionAndOrientation(self.id, position, orientation)

    def get_pose(self):
        return p.getBasePositionAndOrientation(self.id)

    def set_position(self, position):
        _, orn = self.get_pose()
        self.set_pose(position, orn)

    def is_stationary(self, velocity_tolerance=5e-4,
                      angular_velocity_tolerance=5e-4):
        velocity, angular_velocity = tuple(
            p.getBaseVelocity(bodyUniqueId=self.id))
        return (abs(array(velocity)) < velocity_tolerance).all() \
            and (abs(array(angular_velocity))
                 < angular_velocity_tolerance).all()

    def rotate(self, angle_x, angle_y, angle_z):
        position, _ = self.get_pose()
        self.set_pose(
            position,
            p.getQuaternionFromEuler((angle_x, angle_y, angle_z))
        )

    def check_collision(self, other_id=None, collision_distance=0.0):
        others_id = [other_id]
        if other_id is None:
            others_id = [p.getBodyUniqueId(i)
                         for i in range(p.getNumBodies())
                         if p.getBodyUniqueId(i) != self.id]
        for other_id in others_id:
            if len(p.getClosestPoints(
                bodyA=self.id,
                bodyB=other_id,
                    distance=collision_distance)) != 0:
                return True
        return False

    def get_link_idx(self, name):
        joints_info = [
            p.getJointInfo(self.id, joint_idx)
            for joint_idx in range(p.getNumJoints(self.id))
        ]
        for joint_info in joints_info:
            if joint_info[12].decode("utf-8") == name:
                return joint_info[0]
        return -1
