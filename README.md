# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project

A small Graph Attention Network (GAT) to schedule jobs in a ETO manufacturing environement, trained with the Proxmimal Policy gradient Optimization (PPO) reinforcement learning algorithm.

## THIS REPOSITORY IS STILL A WORK IN PROGRESS!!! NOT A FINAL PROJECT!

## Locally try the project
1. `python -m venv ./venv`
2. `source ./venv/bin/activate`
3. `pip install --upgrade -r requirements.txt`
4. `deactivate`

## Architecture of the project
* `/FJS/` contains the code for the traditional Flexible Job Shop scheduling problem;
    * `/FJS/instance_generator.py [nb train instances] [nb test instances] [min jobs] [max jobs] [min resources] [max type of resources] [max resources by type] [max operations]` the code to randomly generate instances and save them into `/FJS/instances/test/` and `/FJS/instances/train/`;
        * e.g. `python FJS/instance_generator.py 10 5 2 10 3 7 3 6`;

    * `/FJS/exact_solver.py [type]` the code to solve the instances using [Google OR solver](https://developers.google.com/optimization);
        * e.g. `python FJS/exact_solver.py test`;

    * `/FJS/gns_solver.py` the code to solve the instances using a Graph Attention Network (GAT) trained using PPO RL;

    * Both `/FJS/instances/test/` and `/FJS/instances/train/` folders contains a `optimal.csv` file with all the optimal values computer by the exact solver; 

* `/EPSIII/` contains the code for third version of the ETO Project Scheduling problem as published by [Neumann et al. (2023)](https://doi.org/10.1016/j.ijpe.2023.109077); 

* `common.py` contains the common code used by several files.

## Typical PKL file with a FJS instance 
```python
FJS_instance = {
    "size": 8, # total number of operations
    "resources": [0, 3, 2, 7], # e.g. 3 resources of type 1; 7 resources of type 3
    "jobs": [  
        [(0, 2), (3, 8), (1, 3)],  # e.g. operation 1 (of job 0) runs on resource type 3 with a processing time of 8 
        [(0, 8), (2, 10), (1, 2)],  
        [(0, 9), (2, 1)]  
]}
```

## Typical translation into a graph structure 
```python
# Instance = {'resources': [1, 2, 2], 'jobs': [[(0, 11), (2, 6), (2, 14), (0, 15), (1, 1), (2, 13)], [(0, 5), (1, 14), (2, 1), (0, 6)]], 'size': 10, 'nb_res': 5}
Graph_overview: HeteroData(
  operation={x=[12, 6]},
  resource={x=[5, 3]},
  (operation, precedence, operation)={edge_index=[2, 12]},
  (operation, uses, resource)={edge_index=[2, 16]},
  (resource, execute, operation)={edge_index=[2, 16]})
Resources: {'x': tensor([[0, 4, 0],
        [0, 2, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 4, 0]])}
Operations: {'x': tensor([[ 0,  0,  0,  0,  0,  0],
        [ 0,  1, 11,  0,  6, 60],
        [ 0,  2,  6, 11,  6, 60],
        [ 0,  2, 14, 17,  6, 60],
        [ 0,  1, 15, 31,  6, 60],
        [ 0,  2,  1, 46,  6, 60],
        [ 0,  2, 13, 47,  6, 60],
        [ 0,  1,  5,  0,  4, 26],
        [ 0,  2, 14,  5,  4, 26],
        [ 0,  2,  1, 19,  4, 26],
        [ 0,  1,  6, 20,  4, 26],
        [ 0,  0,  0,  0,  0,  0]])}
Precedence_relations: {'edge_index': tensor([[ 1,  2,  3,  4,  5,  7,  8,  9,  0,  6,  0, 10],
        [ 2,  3,  4,  5,  6,  8,  9, 10,  1, 11,  7, 11]])}
Requirements operation->resource: {'edge_index': tensor([[ 1,  2,  2,  3,  3,  4,  5,  5,  6,  6,  7,  8,  8,  9,  9, 10],
        [ 0,  3,  4,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0]])}
Requirements resource->operation: {'edge_index': tensor([[ 0,  3,  4,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0],
        [ 1,  2,  2,  3,  3,  4,  5,  5,  6,  6,  7,  8,  8,  9,  9, 10]])}
```