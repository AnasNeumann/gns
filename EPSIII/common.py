import pickle
import os
import resource

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
    return [[default_value] * a for _ in range(b)]

def init_3D(a, b, c, default_value):
    return [[[default_value] * a for _ in range(b)] for _ in range(c)]

def init_several_1D(a, default_value, nb):
    return (init_1D(a, default_value) for _ in range(nb))

def init_several_2D(a, b, default_value, nb):
    return (init_2D(a, b, default_value) for _ in range(nb))

def init_several_3D(a, b, c, default_value, nb):
    return (init_3D(a, b, c, default_value) for _ in range(nb))