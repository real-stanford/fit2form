from .GraspSimulationEnv import GraspSimulationEnv
from .GraspObjectGenerationEnv import GraspObjectGenerationEnv
from .ImprintGenerationEnv import ImprintGenerationEnv
import ray


GraspSimulationEnv = ray.remote(GraspSimulationEnv)
GraspObjectGenerationEnv = ray.remote(GraspObjectGenerationEnv)
ImprintGenerationEnv = ray.remote(ImprintGenerationEnv)

__all__ = [
    'GraspSimulationEnv',
    'ImprintGenerationEnv',

]
