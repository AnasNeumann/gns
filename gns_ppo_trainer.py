import pickle
import os
from model.instance import Instance
from common import directory
import torch
torch.autograd.set_detect_anomaly(True)
import random
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from debug.debug_gns import debug_printer
from typing import Callable
from model.agent import MultiAgent_OneInstance, MultiAgents_Batch

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

LEARNING_RATE = 2e-4
PROBLEM_SIZES = [['s', 'm'], ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']]
PPO_CONF = {
    "validation_rate": 10,
    "switch_batch": 20,
    "train_iterations": [10, 1000], 
    "opt_epochs": 3,
    "batch_size": [3, 20],
    "clip_ratio": 0.2,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation_ratio': 0.1
}
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

def reward(makespan_old: int, makespan_new: int, cost_old: int=-1, cost_new: int=-1, a: float=-1, use_cost: bool=False):
    if use_cost:
        return a * (cost_old - cost_new) + (1-a) * (makespan_old - makespan_new)
    else:
        return makespan_old - makespan_new

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str):
    complete_path = path + directory.models
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights.pth')

def search_instance(instances: list[Instance], id: int) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(debug_mode: bool, path: str):
    instances = [] 
    for size in PROBLEM_SIZES[0 if debug_mode else 1]:
        complete_path = path+directory.instances+'/train/'+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def train_or_validate_batch(agents: list[(Module, str)], batch: list[Instance],train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str, debug: bool):
    instances_results: list[MultiAgent_OneInstance] = []
    for instance in batch:
        print(f"\t start solving instance: {instance.id}...")
        instances_results.append(solve_function(instance, agents, path="", train=True, device=device, debug_mode=debug))
    batch_result: MultiAgents_Batch = MultiAgents_Batch(
        batch=instances_results, 
        agent_names=[name for _,name in agents], 
        gamma=PPO_CONF['discount_factor'], 
        lam=PPO_CONF['bias_variance_tradeoff'],
        weight_policy_loss=PPO_CONF['policy_loss'],
        weight_value_loss=PPO_CONF['value_loss'], 
        weight_entropy_bonus=PPO_CONF['entropy'],
        clipping_ratio=PPO_CONF['clip_ratio'])
    if train:
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            optimizer.zero_grad()
            loss: Tensor = batch_result.compute_losses(agents)
            print(f"\t Multi-agent batch loss: {loss} - Differentiable computation graph = {loss.requires_grad}!")
            loss.backward(retain_graph=False)
            optimizer.step()
    else:
        loss: Tensor = batch_result.compute_losses(agents)
        print(f'\t Multi-agent batch loss: {loss:.4f}')

def PPO_train(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, debug_mode: bool=False):
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations: int =PPO_CONF['train_iterations'][0 if debug_mode else 1]
    batch_size: int =PPO_CONF['batch_size'][0 if debug_mode else 1]
    epochs: int =PPO_CONF['opt_epochs']
    debug_print: Callable = debug_printer(debug_mode)
    instances: list[Instance] = load_training_dataset(path=path, debug_mode=debug_mode)
    embedding_stack.train()
    embedding_stack.to(device)
    shared_critic.train()
    shared_critic.to(device)
    for agent,_ in agents:
        agent.train()
        agent.to(device)
    optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[OUTSOURCING][AGENT].parameters()), 
        lr=LEARNING_RATE
    )
    random.shuffle(instances)
    num_val = int(len(instances) * PPO_CONF['validation_ratio'])
    train_instances, val_instances = instances[num_val:], instances[:num_val]
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % PPO_CONF['switch_batch'] == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch = random.sample(train_instances, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, optimizer=optimizer, solve_function=solve_function, device=device, debug=debug_mode)
        if iteration % PPO_CONF['validation_rate'] == 0:
            debug_print("\t time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            embedding_stack.eval()
            with torch.no_grad():
                train_or_validate_batch(agents, val_instances, train=False, epochs=-1, optimizer=None, solve_function=solve_function, device=device, debug=debug_mode)
            for agent,_ in agents:
                agent.train()
            embedding_stack.train()
    save_models(agents, embedding_stack, shared_critic, path=path)
    print("<======***--| END OF TRAINING |--***======>")