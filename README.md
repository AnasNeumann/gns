# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project

A small Graph Attention Network (GAT) to schedule jobs in a ETO manufacturing environement, trained with the Proxmimal Policy gradient Optimization (PPO) reinforcement learning algorithm.

## Locally try the project
1. `python -m venv ./venv`
2. `source ./venv/bin/activate`
3. `pip install --upgrade -r requirements.txt`
4. `deactivate`

## Architecture of the project
* `/FJS/` contains the code for the traditional Flexible Job Shop scheduling problem;
    * `/FJS/instance_generator.py` the code to randomly generate instances and save them into `/FJS/instances/`;
    * `/FJS/exact_solver.py` the code to solve the instances using [Google OR solver](https://developers.google.com/optimization);
    * `/FJS/gns.py` the code to solve the instances using a Graph Attention Network (GAT) and PPO. 
* `/EPSIII/` contains the code for third version of the ETO Project Scheduling problem as published by [Neumann et al. (2023)](https://doi.org/10.1016/j.ijpe.2023.109077); 
* `common.py` contains the common code used by several files.