import pickle
import os
import resource
from typing import Union, Any, Dict
from torch.nn import Module

# =====================================================================
# =*= COMMON TOOLS (objects and functions) USED ACCROSS THE PROJECT =*=
# =====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class ProjectDirectory:
    def __init__(self, data: str, instances: str, models: str, results: str, out: str, scripts: str, solutions: str):
        self.data: str = data
        self.instances: str = self.data+'/'+instances
        self.models: str = self.data+'/'+models
        self.out: str = self.data+'/'+out
        self.results: str = self.data+'/'+results
        self.solutions: str = self.data+'/'+solutions
        self.scripts: str = scripts

directory = ProjectDirectory('data', 'instances', 'models', 'results', 'out', 'jobs/scripts', 'solutions')

num_feature = Union[int, float]
all_types_feature = Union[int, float, bool, list]
generic_object = Union[object, Dict[Any, Any]]

def objective_value(cmax: int, cost: int, cmax_weight: float):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def load_instance(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_instances(path: str):
    print(f"Loading data from path: {path}...")
    instances = []
    for i in os.listdir(path):
        if i.endswith('.pkl'):
            file_path = os.path.join(path, i)
            with open(file_path, 'rb') as file:
                instances.append(pickle.load(file))
    print("end of loading!")
    return instances

def set_memory_limit(max_ram_bytes: num_feature):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_ram_bytes * 1024 * 1024 * 1024, hard))

def init_1D(a: int, default_value: all_types_feature):
    return [default_value] * a

def init_2D(a: int, b: int, default_value: all_types_feature):
    return [[default_value for _ in range(b)] for _ in range(a)]

def init_3D(a: int, b: int, c: int, default_value: all_types_feature):
    return [[[default_value for _ in range(c)] for _ in range(b)] for _ in range(a)]

def init_several_1D(a: int, default_value: all_types_feature, nb: int):
    return (init_1D(a, default_value) for _ in range(nb))

def init_several_2D(a: int, b: int, default_value: all_types_feature, nb: int):
    return (init_2D(a, b, default_value) for _ in range(nb))

def init_several_3D(a: int, b: int, c: int, default_value: all_types_feature, nb: int):
    return (init_3D(a, b, c, default_value) for _ in range(nb))

def to_bool(s: str):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

def search_object_by_id(objects: list[generic_object], id: int):
    for obj in objects:
        if obj['id'] == id:
            return obj
    return None

def freeze(agent: Module):
    for param in agent.parameters():
        param.requires_grad = False

def unfreeze(agent: Module):
    agent.train()
    for param in agent.parameters():
        param.requires_grad = True

def freeze_several_and_unfreeze_others(agents: list[(Module, str)], to_freeze: list[str]):
    for agent, name in agents:
        if name in to_freeze:
            freeze(agent)
    for agent, name in agents:
        if name not in to_freeze:
            unfreeze(agent) 

def unfreeze_all(agents: list[(Module, str)]):
    for agent, _ in agents:
        unfreeze(agent)