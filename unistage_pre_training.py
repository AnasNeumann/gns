import pickle
import os
from model.instance import Instance
from tools.common import directory, freeze, unfreeze, unfreeze_all, freeze_several_and_unfreeze_others
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
from debug.loss_tracker import LossTracker
from model.reward_memory import Memories, Memory
from torch.optim import Adam

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PROBLEM_SIZES = [['s'], ['s', 'm'], ['s', 'm', 'l'], ['s', 'm', 'l', 'xl'], ['s', 'm', 'l', 'xl', 'xxl']] # ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
PPO_CONF = {
    "validation_rate": 20,
    "switch_batch": 10,
    "train_iterations": 500,
    "opt_epochs": 3,
    "batch_size": 20,
    "clip_ratio": 0.1,
    "policy_loss": 1.0,
    'validation': 10,
    "value_loss": 0.1,
    "entropy": 0.1,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0
}
AGENTS = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, memory: Memories, run_number:int, complete_path: str):
    index = str(run_number)
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights_'+index+'.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights_'+index+'.pth')
    torch.save(optimizer.state_dict(), complete_path+'/adam_weights_'+index+'.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights_'+index+'.pth')
    with open(complete_path+'/memory_'+index+'.pth', 'wb') as f:
            pickle.dump(memory, f)

def search_instance(instances: list[Instance], id: int) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(run_time: int, path: str, train: bool = True):
    type: str = '/train/' if train else '/test/'
    instances = []
    for size in PROBLEM_SIZES[run_time]:
        complete_path = path+directory.instances+type+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def train_or_validate_batch(reward_MEMORIES: Memories, agents: list[(Module, str)], batch: list[Instance], agent_names: list[str], train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str):
    """
        Train or validate on a batch of instances
    """
    _batch_replay_memory: list[MultiAgent_OneInstance] = []
    for instance in batch:
        reward_MEMORY: Memory = reward_MEMORIES.add_instance_if_new(instance_id=instance.size + "_" + str(instance.id))
        print(f"\t start solving instance: {instance.id}...")
        transitions,_,_,_,_ = solve_function(instance=instance, agents=agents, device=device, train=True, trainable=[1,1,1], reward_MEMORY=reward_MEMORY)
        _batch_replay_memory.append(transitions)
    batch_result: MultiAgents_Batch = MultiAgents_Batch(
        batch=_batch_replay_memory, 
        agent_names=agent_names, 
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

def multi_agent_stage(train_data: list[Instance], val_data: list[Instance], agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Adam, reward_MEMORIES: Memories, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, switch_batch: int, validation_rate: int, device: str, run_number: int):
    """
        Last (4th) optimization stage: all three agents together
    """
    vlosses = MAPPO_Losses(agent_names=[name for _,name in agents])
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % switch_batch == 0:
            print(f"\t New training batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        random.shuffle(current_batch)
        train_or_validate_batch(reward_MEMORIES=reward_MEMORIES, agents=agents, batch=current_batch, train=True, epochs=epochs, optimizer=optimizer, agent_names=[name for _,name in agents], solve_function=solve_function, device=device)
        if iteration % validation_rate == 0:
            print("\t Validation stage...")
            for agent,_ in agents:
                agent.eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(reward_MEMORIES=reward_MEMORIES, agents=agents, batch=val_data, train=False, epochs=-1, agent_names=[name for _,name in agents], optimizer=None, solve_function=solve_function, device=device)
                vlosses.add(current_vloss)
            for agent,_ in agents:
                agent.train()
    complete_path = path + directory.models
    with open(complete_path+'/validation_'+str(run_number)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=optimizer, memory=reward_MEMORIES, run_number=run_number, complete_path=complete_path)

def uni_stage_pre_train(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Adam, memory: Memories, path: str, solve_function: Callable, device: str, run_number:int):
    """
        Multi-stage PPO function to pre-train agents on several instances
    """
    _start_time = systime.time()
    print("Loading dataset....")
    instances: list[Instance] = load_training_dataset(path=path, train=True, run_time=run_number-1)
    print(f"Dataset loaded after {(systime.time()-_start_time)} seconds!")
    random.shuffle(instances)
    _num_val = PPO_CONF['validation']
    train_data, val_data = instances[_num_val:], instances[:_num_val]
    embedding_stack = embedding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for agent,_ in agents:
        agent = agent.to(device)
        agent.train()
    multi_agent_stage(train_data=train_data,
            val_data=val_data,
            agents=agents, 
            embedding_stack=embedding_stack,
            shared_critic=shared_critic,
            optimizer=optimizer,
            reward_MEMORIES=memory,
            solve_function=solve_function, 
            path=path, 
            epochs=PPO_CONF['opt_epochs'], 
            iterations=PPO_CONF['train_iterations'],
            batch_size=PPO_CONF['batch_size'], 
            switch_batch=PPO_CONF['switch_batch'],
            validation_rate=PPO_CONF['validation_rate'],
            device=device, 
            run_number=run_number)