import pickle
import os
from model.instance import Instance
from tools.common import directory
import torch
torch.autograd.set_detect_anomaly(True)
import random
import pandas as pd
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from debug.debug_gns import debug_printer
from typing import Callable
from model.agent import MultiAgent_OneInstance, MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses
import time as systime
from tools.common import load_instance
from translators.graph2solution_translator import translate_solution
from debug.loss_tracker import LossTracker
import math

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
    "train_iterations": {
        "pre_training": [2, 150], 
        "fine_tuning": [2, 150]
    },
    "opt_epochs": 3,
    "batch_size": {
        "pre_training": [2, 15],
        "fine_tuning": [2, 10],
    },
    "clip_ratio": 0.2,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation': 10
}
AGENTS = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
small_steps: float = 0.3
big_steps: float = 0.7

def reward(makespan_old: int, makespan_new: int, last_op_old: int, last_op_new: int, cost_old: int=-1, cost_new: int=-1, a: float=-1, use_cost: bool=False):
    if use_cost:
        return  a * (big_steps * (makespan_old - makespan_new) + small_steps * (last_op_old - last_op_new)) + (1-a) * (cost_old - cost_new)
    else:
        return a * (big_steps * makespan_old - makespan_new + small_steps * (last_op_old - last_op_new))

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

def objective_value(cmax: int, cost: int, cmax_weight: float):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def train_or_validate_batch(agents: list[(Module, str)], batch: list[Instance], train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str, debug: bool):
    """
        Train or validate on a batch of instances
    """
    instances_results: list[MultiAgent_OneInstance] = []
    for instance in batch:
        print(f"\t start solving instance: {instance.id}...")
        r,_,_,_ = solve_function(instance=instance, agents=agents, train=True, trainable=[True for _ in agents], device=device, debug_mode=debug)
        instances_results.append(r)
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

def PPO_pre_train(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, path: str, solve_function: Callable, device: str, run_number:int, debug_mode: bool=False):
    """
        PPO function to pre-train agents on several instances
    """
    start_time = systime.time()
    iterations: int = PPO_CONF['train_iterations']['pre_training'][0 if debug_mode else 1]
    batch_size: int = PPO_CONF['batch_size']['pre_training'][0 if debug_mode else 1]
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
    print("<======***--| END OF PRE-TRAINING |--***======>")

def PPO_fine_tuning(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, path: str, solve_function: Callable, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        PPO function to fine-tune agents on the target instance
    """
    start_time = systime.time()
    iterations: int = PPO_CONF['train_iterations']['fine_tuning'][0 if debug_mode else 1]
    epochs: int = PPO_CONF['opt_epochs']
    target_instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    print(f"Target instance {size}_{id} loaded....")
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()
    losses = MAPPO_Losses(agent_names=[name for _,name in agents])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per solving episode)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per solving episode)", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per solving episode)", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="pink", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _cost_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Cost", title="Final Cost by episode", color="orange", show=interactive)
    _best_obj: float = math.inf
    _best_episode: int = 0
    _time_to_best: float = 0
    for episode in range(iterations):
        print(f"PPO episode: {episode+1}/{iterations}:")
        loss, graph, cmax, cost = solve_function(instance=target_instance, agents=agents, train=True, trainable=[True for _ in agents], device=device, debug_mode=debug_mode)
        _current_obj = objective_value(cmax, cost, target_instance.w_makespan)
        if _current_obj < _best_obj:
            _best_obj = _current_obj
            _best_episode = episode
            _time_to_best = systime.time()-start_time
        _Cmax_TRACKER.update(cmax)
        _cost_TRACKER.update(cost)
        batch_result: MultiAgents_Batch = MultiAgents_Batch(
                batch=[loss],
                agent_names=[name for _,name in agents],
                gamma=PPO_CONF['discount_factor'],
                lam=PPO_CONF['bias_variance_tradeoff'],
                weight_policy_loss=PPO_CONF['policy_loss'],
                weight_value_loss=PPO_CONF['value_loss'], 
                weight_entropy_bonus=PPO_CONF['entropy'],
                clipping_ratio=PPO_CONF['clip_ratio'])
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            optimizer.zero_grad()
            training_loss, details = batch_result.compute_losses(agents, return_details=True)
            print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
            training_loss.backward(retain_graph=False)
            optimizer.step()
            details: MAPPO_Loss
            _vloss_TRACKER.update(details.value_loss)
            _scheduling_loss_TRACKER.update(details.get(AGENTS[SCHEDULING]).policy_loss)
            _outsourcing_loss_TRACKER.update(details.get(AGENTS[OUTSOURCING]).policy_loss)
            losses.add(details)
    print("<======***--| END OF FINE-TUNING |--***======>")

    final_metrics = pd.DataFrame({
        'index': [target_instance.id],
        'value': [_best_obj],
        'episode': [_best_episode],
        'time_to_best': [_time_to_best],
        'computing_time': [systime.time()-start_time],
        'device_used': [device]})
    _vloss_TRACKER.save(directory.solutions+'/'+size+'/value_loss_'+id)
    _scheduling_loss_TRACKER.save(directory.solutions+'/'+size+'/scheduling_loss_'+id)
    _outsourcing_loss_TRACKER.save(directory.solutions+'/'+size+'/outsourcing_loss_'+id)
    _cost_TRACKER.save(directory.solutions+'/'+size+'/final_cost_'+id)
    _Cmax_TRACKER.save(directory.solutions+'/'+size+'/final_cmax_'+id)
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_fine_tuned_gns_'+id+'.csv', index=False)
    with open(directory.solutions+'/'+size+'/fine_tune_gns_graph_'+id+'.pkl', 'wb') as f:
        pickle.dump(graph, f)
    with open(directory.solutions+'/'+size+'/fine_tune_gns_solution_'+id+'.pkl', 'wb') as f:
        pickle.dump(translate_solution(graph, target_instance), f)
    with open(directory.solutions+'/'+size+'/fine_tune_losses_'+id+'.pkl', 'wb') as f:
        pickle.dump(losses, f)
    print("<======***--| ALL RESULTS SAVED |--***======>")
    
    