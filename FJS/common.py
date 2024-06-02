import pickle
import os

# Common definition
OP_STRUCT = {"resource_type": 0, "duration": 1}

# Function to load instances 
def load_instances(path):
    instances = []
    for i in os.listdir(path):
        if i.endswith('.pkl'):
            file_path = os.path.join(path, i)
            print(f"Loading data from: {i}...")
            with open(file_path, 'rb') as file:
                instances.append(pickle.load(file))
    return instances

def shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return tuple(shape)