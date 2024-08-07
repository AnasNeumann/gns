import pickle
import os
import resource
import torch

def load_instance(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_instances(path):
    print(f"Loading data from path: {path}...")
    instances = []
    for i in os.listdir(path):
        if i.endswith('.pkl'):
            file_path = os.path.join(path, i)
            with open(file_path, 'rb') as file:
                instances.append(pickle.load(file))
    print("end of loading!")
    return instances

def set_memory_limit(max_ram_bytes):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_ram_bytes * 1024 * 1024 * 1024, hard))

def init_1D(a, default_value):
    return [default_value] * a

def init_2D(a, b, default_value):
    return [[default_value for _ in range(b)] for _ in range(a)]

def init_3D(a, b, c, default_value):
    return [[[default_value for _ in range(c)] for _ in range(b)] for _ in range(a)]

def init_several_1D(a, default_value, nb):
    return (init_1D(a, default_value) for _ in range(nb))

def init_several_2D(a, b, default_value, nb):
    return (init_2D(a, b, default_value) for _ in range(nb))

def init_several_3D(a, b, c, default_value, nb):
    return (init_3D(a, b, c, default_value) for _ in range(nb))

def to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

def features2tensor(features):
    return torch.tensor([[f for f in features]], dtype=torch.float)

def id2tensor(id1, id2):
    return torch.tensor([[id1], [id2]], dtype=torch.long)