import pickle
import os
from model.instance import Instance
from tools.common import directory
import torch
torch.autograd.set_detect_anomaly(True)
import random
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from debug.debug_gns import debug_printer
from typing import Callable
from model.agent import MultiAgent_OneInstance, MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses
import time as systime

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PROBLEM_SIZES = [['s', 'm'], ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']]
PPO_CONF = {
    "validation_rate": 20,
    "switch_batch": 10,
    "train_iterations": [2, 150], 
    "opt_epochs": 3,
    "batch_size": [2, 15],
    "clip_ratio": 0.2,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation': 10
}
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

def reward(makespan_old: int, makespan_new: int, cost_old: int=-1, cost_new: int=-1, a: float=-1, use_cost: bool=False):
    if use_cost:
        return a * (cost_old - cost_new) + (1-a) * (makespan_old - makespan_new)
    else:
        return a * (makespan_old - makespan_new)

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, run_number:int, complete_path: str):
    index = str(run_number)
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights_'+index+'.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights_'+index+'.pth')
    torch.save(optimizer.state_dict(), complete_path+'/adam_'+index+'.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights_'+index+'.pth')

def search_instance(instances: list[Instance], id: int) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(debug_mode: bool, path: str, train: bool = True):
    type: str = '/train/' if train else '/test/'
    instances = []
    for size in PROBLEM_SIZES[0 if debug_mode else 1]:
        complete_path = path+directory.instances+type+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def train_or_validate_batch(agents: list[(Module, str)], batch: list[Instance], train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str, debug: bool):
    instances_results: list[MultiAgent_OneInstance] = []
    for instance in batch:
        print(f"\t start solving instance: {instance.id}...")
        instances_results.append(solve_function(instance=instance, agents=agents, train=True, device=device, debug_mode=debug))
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
            training_loss: Tensor = batch_result.compute_losses(agents, return_details=False)
            print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
            training_loss.backward(retain_graph=False)
            optimizer.step()
    else:
        current_vloss, current_details = batch_result.compute_losses(agents, return_details=True)
        print(f'\t Multi-agent batch loss: {current_vloss:.4f}')
        return current_details

def PPO_train(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, path: str, solve_function: Callable, device: str, run_number:int, debug_mode: bool=False):
    start_time = systime.time()
    iterations: int = PPO_CONF['train_iterations'][0 if debug_mode else 1]
    batch_size: int = PPO_CONF['batch_size'][0 if debug_mode else 1]
    epochs: int = PPO_CONF['opt_epochs']
    debug_print: Callable = debug_printer(debug_mode)
    print("Loading dataset....")
    instances: list[Instance] = load_training_dataset(path=path, train=True, debug_mode=debug_mode)
    print(f"Dataset loaded after {(systime.time()-start_time)} seconds!")
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()
    vlosses = MAPPO_Losses(agent_names=[name for _,name in agents])
    random.shuffle(instances)
    num_val = PPO_CONF['validation']
    train_data, val_data = instances[num_val:], instances[:num_val]
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % PPO_CONF['switch_batch'] == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, optimizer=optimizer, solve_function=solve_function, device=device, debug=debug_mode)
        if iteration % PPO_CONF['validation_rate'] == 0:
            debug_print("\t time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            embedding_stack.eval()
            shared_critic.eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents, val_data, train=False, epochs=-1, optimizer=None, solve_function=solve_function, device=device, debug=debug_mode)
                vlosses.add(current_vloss)
            for agent,_ in agents:
                agent.train()
            embedding_stack.train()
            embedding_stack.train()
    complete_path = path + directory.models
    with open(complete_path+'/validation_'+str(run_number)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=optimizer, run_number=run_number, complete_path=complete_path)
    print("<======***--| END OF TRAINING |--***======>")