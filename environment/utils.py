import pybullet as p


def wait_until_object_is_stable(
        grasp_object,
        max_wait_steps=int(1e5),
        step_simulation_fn=None):
    if step_simulation_fn is None:
        step_simulation_fn = p.stepSimulation
    for _ in range(50):
        step_simulation_fn()
    for _ in range(max_wait_steps):
        step_simulation_fn()
        if grasp_object.is_stationary():
            return True
    return False
