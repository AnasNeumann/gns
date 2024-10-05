import pickle
from torch.multiprocessing import Pool
import os
from model.instance import Instance
from model.graph import State
from common import init_several_1D, search_object_by_id, generic_object, directory
import torch
torch.autograd.set_detect_anomaly(True)
import random
import multiprocessing
from typing import Tuple
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from debug.debug_gns import debug_printer
from typing import Callable
import numpy as np 
from model.agent import MultiAgent_OneInstance, MultiAgents_Batch

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

LEARNING_RATE = 2e-4
PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
PPO_CONF = {
    "validation_rate": 10,
    "switch_batch": 20,
    "train_iterations": [10, 1000], 
    "opt_epochs": 3,
    "batch_size": [4, 20],
    "clip_ratio": 0.2,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation_ratio': 0.1
}

def reward(makespan_old: int, makespan_new: int, cost_old: int=-1, cost_new: int=-1, a: float=-1, use_cost: bool=False):
    if use_cost:
        return a * (cost_old - cost_new) + (1-a) * (makespan_old - makespan_new)
    else:
        return makespan_old - makespan_new

def save_models(agents: list[(Module, str)], path):
    complete_path = path + directory.models
    torch.save(agents[0].shared_embedding_layers.state_dict(), complete_path+'/gnn_weights.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights.pth')

def search_instance(instances: list[Instance], id: int) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(debug_mode: bool, path: str):
    instances = [] 
    for size in PROBLEM_SIZES if not debug_mode else ['m']:
        complete_path = path+directory.instances+'/train/'+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def optimize(optimizer: Optimizer, loss: float):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

def async_solve_one(init_args: Tuple[list, Instance, bool, Callable]):
    agents, instance, debug_mode, solve_function = init_args
    print(f"\t start solving instance: {instance.id}...")
    result = solve_function(instance, agents, train=True, debug_mode=debug_mode)
    print(f"\t end solving instance: {instance.id}!")
    return result

def async_solve_batch(agents: list[(Module, str)], batch: list[Instance], num_processes: int, train: bool, epochs: int, optimizers: list[Optimizer], solve_function: Callable, debug: bool):
    with Pool(num_processes) as pool:
        instances_results: list[MultiAgent_OneInstance] = pool.map(async_solve_one, [(agents, instance, debug, solve_function) for instance in batch])
    batch_result: MultiAgents_Batch = MultiAgents_Batch(
        batch=instances_results, 
        agent_names=[name for _,name in agents], 
        gamma=PPO_CONF['discount_factor'], 
        lam=PPO_CONF['bias_variance_tradeoff'],
        weight_policy_loss=PPO_CONF['policy_loss'],
        weight_value_loss=PPO_CONF['value_loss'], 
        weight_entropy_bonus=PPO_CONF['entropy'],
        clipping_ratio=PPO_CONF['clip_ratio'])
    if train and epochs>0:
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            losses = batch_result.compute_losses(agents)
            for agent_id, (_, name) in enumerate(agents):
                print(f"\t\t Optimizing agent: {name}...")
                optimize(optimizers[agent_id], losses[agent_id])
    else:
        losses = batch_result.compute_losses(agents)
        for agent_id, (_, name) in enumerate(agents):
            print(f'\t Average loss of agent {name} = {losses[agent_id]:.4f}')

def PPO_train(agents: list[(Module, str)], path: str, solve_function: Callable, debug_mode: bool=False):
    device = "cuda" if not debug_mode and torch.cuda.is_available() else "cpu"
    iterations=PPO_CONF['train_iterations'][0 if debug_mode else 1]
    batch_size=PPO_CONF['batch_size'][0 if debug_mode else 1]
    epochs=PPO_CONF['opt_epochs']
    validation_rate=PPO_CONF['validation_rate'],
    debug_print = debug_printer(debug_mode)
    instances = load_training_dataset(path=path, debug_mode=debug_mode)
    optimizers = [torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE) for agent,_ in agents]
    for agent,_ in agents:
        agent.train()
        if device == "cuda":
            agent.to(device)
    random.shuffle(instances)
    num_val = int(len(instances) * PPO_CONF['validation_ratio'])
    train_instances, val_instances = instances[num_val:], instances[:num_val]
    num_processes = multiprocessing.cpu_count() if not debug_mode else 1
    print(f"Start training models with PPO running on {num_processes} TPUs in parallel...")
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % PPO_CONF['switch_batch'] == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch = random.sample(train_instances, batch_size)
        async_solve_batch(agents, current_batch, num_processes, train=True, epochs=epochs, optimizers=optimizers, solve_function=solve_function, debug=debug_mode)
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            with torch.no_grad():
                async_solve_batch(agents, val_instances, num_processes, train=False, epochs=-1, optimizers=[], debug=debug_mode)
            for agent,_ in agents:
                agent.train()
    save_models(agents, path=path)
    print("<======***--| END OF TRAINING |--***======>")