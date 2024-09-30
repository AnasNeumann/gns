import pickle
import os
import resource
import torch
from typing import Union, Any, Dict

num_feature = Union[int, float]
all_types_feature = Union[int, float, bool, list]
generic_object = Union[object, Dict[Any, Any]]
      
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

def features2tensor(features: list):
    for f in features:
        if isinstance(f, bool):
            f = to_binary(f)
    return torch.tensor([[f for f in features]], dtype=torch.float)

def id2tensor(id1: int, id2: int):
    return torch.tensor([[id1], [id2]], dtype=torch.long)

def search_object_by_id(objects: list[generic_object], id: int):
    for obj in objects:
        if obj['id'] == id:
            return obj
    return None