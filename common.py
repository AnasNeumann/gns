import pickle
import os
import resource
import torch
from typing import Union, Any, Dict
from model.instance import Instance
from torch import Tensor

# =====================================================================
# =*= COMMON TOOLS (objects and functions) USED ACCROSS THE PROJECT =*=
# =====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class ProjectDirectory:
    def __init__(self, data: str, instances: str, models: str, out: str, scripts: str):
        self.data: str = data
        self.instances: str = self.data+'/'+instances
        self.models: str = self.data+'/'+models
        self.out: str = self.data+'/'+out
        self.scripts: str = scripts

directory = ProjectDirectory('data', 'instances', 'models', 'out', 'jobs/scripts')

num_feature = Union[int, float]
all_types_feature = Union[int, float, bool, list]
generic_object = Union[object, Dict[Any, Any]]

def add_into_tensor(tensor_list: Tensor | None, tensor_val: Tensor):
    if tensor_list is None:
        tensor_list = tensor_val
    else:
        tensor_list = torch.cat((tensor_list, tensor_val), dim=0)
    return tensor_list

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

def to_binary(booleanVal: bool):
    return 1 if booleanVal else 0

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

def features2tensor(features: list, device: str):
    for f in features:
        if isinstance(f, bool):
            f = to_binary(f)
    return torch.tensor([[f for f in features]], dtype=torch.float, device=device)

def id2tensor(id1: int, id2: int, device: str):
    return torch.tensor([[id1], [id2]], dtype=torch.long, device=device)

def search_object_by_id(objects: list[generic_object], id: int):
    for obj in objects:
        if obj['id'] == id:
            return obj
    return None