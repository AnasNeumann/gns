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
        "size": 8 # total number of operations
        "resources": [0, 3, 2, 7], # e.g. 3 resources of type 1; 7 resources of type 3
        "jobs": [  
            [(0, 2), (3, 8), (1, 3)],  # e.g. operation 1 (of job 0) runs on resource type 3 with a processing time of 8 
            [(0, 8), (2, 10), (1, 2)],  
            [(0, 9), (2, 1)]  
    ]}
```

## Typical translation into a graph structure 
```python
        # Instance = {'resources': [1, 2, 3, 2], 'jobs': [[(2, 1), (2, 12), (2, 14), (3, 9), (0, 6)], [(1, 14), (1, 1)], [(3, 1), (1, 2), (1, 2), (1, 7), (3, 12)], [(3, 4), (1, 11), (0, 6)], [(1, 15), (1, 2)]], 'size': 17}
        Graph_overview: HeteroData(
                operation={x=[19, 6]},
                resource={x=[8, 3]},
                (operation, precedence, operation)={edge_index=[2, 12]},
                (start, to, operation)={edge_index=[2, 5]},
                (operation, to, end)={edge_index=[2, 5]},
                (operation, uses, resource)={edge_index=[2, 35]})
        Resources: {'x': tensor([[0, 2, 0],
                [0, 8, 0],
                [0, 8, 0],
                [0, 3, 0],
                [0, 3, 0],
                [0, 3, 0],
                [0, 4, 0],
                [0, 4, 0]])}
        Operations: {'x': tensor([[ 0,  0,  0,  0,  0,  0],
                [ 0,  3,  1,  0,  5,  0],
                [ 0,  3, 12,  1,  5,  0],
                [ 0,  3, 14, 13,  5,  0],
                [ 0,  2,  9, 27,  5,  0],
                [ 0,  1,  6, 36,  5,  0],
                [ 0,  2, 14,  0,  2,  0],
                [ 0,  2,  1, 14,  2,  0],
                [ 0,  2,  1,  0,  5,  0],
                [ 0,  2,  2,  1,  5,  0],
                [ 0,  2,  2,  3,  5,  0],
                [ 0,  2,  7,  5,  5,  0],
                [ 0,  2, 12, 12,  5,  0],
                [ 0,  2,  4,  0,  3,  0],
                [ 0,  2, 11,  4,  3,  0],
                [ 0,  1,  6, 15,  3,  0],
                [ 0,  2, 15,  0,  2,  0],
                [ 0,  2,  2, 15,  2,  0],
                [ 0,  0,  0,  0,  0,  0]])}
        Precedence_relations: {'edge_index': tensor([[ 1,  2,  3,  4,  6,  8,  9, 10, 11, 13, 14, 16],
                [ 2,  3,  4,  5,  7,  9, 10, 11, 12, 14, 15, 17]])}
        Link_with_dummy_start_node: {'edge_index': tensor([[ 0,  0,  0,  0,  0],
                [ 1,  6,  8, 13, 16]])}
        Link_with_dummy_end_node: {'edge_index': tensor([[ 5,  7, 12, 15, 17],
                [18, 18, 18, 18, 18]])}
        Requirements: {'edge_index': tensor([[ 1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  5,  6,  6,  7,  7,  8,  8,
                9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17],
                [ 3,  4,  5,  3,  4,  5,  3,  4,  5,  6,  7,  0,  1,  2,  1,  2,  6,  7,
                1,  2,  1,  2,  1,  2,  6,  7,  6,  7,  1,  2,  0,  1,  2,  1,  2]])}
```