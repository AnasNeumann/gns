import pickle
import os

# Common definition
OP_STRUCT = {"resource_type": 0, "duration": 1}

# Function to load instances 
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

def shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return tuple(shape)