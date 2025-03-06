import pickle
import os
from model.instance import Instance
from tools.common import directory
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import Callable
from model.agent import MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses, MultiAgent_OneInstance
import time as systime
from tools.common import load_instance, objective_value
from translators.graph2solution_translator import translate_solution
from debug.loss_tracker import LossTracker
import math

# ==================================================================
# =*= MULTI-STAGE PROXIMAL POLICY OPTIMIZATION (PPO) FINE TUNING =*=
# ==================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PROBLEM_SIZES = [['s', 'm'], ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']]
PPO_CONF = {
    "validation_rate": 20,
    "switch_batch": 10,
    "train_iterations": [2, 300],
    "opt_epochs": 3,
    "clip_ratio": 0.1,
    "policy_loss": 1.0,
    "batch": 10,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0
}
LEARNING_RATE = 1e-4
AGENTS = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
small_steps: float = 0.3
big_steps: float = 0.7

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, run_number:int, complete_path: str):
    index = str(run_number)
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights_'+index+'.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights_'+index+'.pth')
    torch.save(optimizer.state_dict(), complete_path+'/adam_'+index+'.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights_'+index+'.pth')

def scheduling_stage(target_instance : Instance, agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        First (1st) optimization stage: scheduling agent alone
    """
    scheduling_optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[SCHEDULING][AGENT].parameters()), 
        lr=LEARNING_RATE)
    embedding_stack.train()
    shared_critic.train()
    _agent, _  = agents[SCHEDULING]
    _agent.train()
    losses = MAPPO_Losses(agent_names=[AGENTS[SCHEDULING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per solving episode)", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    batch_results: list[MultiAgent_OneInstance] = []
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        loss, _, cmax, _ = solve_function(instance=target_instance, agents=agents, train=True, trainable=[0,1,0], device=device, debug_mode=debug_mode)
        batch_results.append(loss)
        _Cmax_TRACKER.update(cmax)
        if episode % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=batch_results,
                    agent_names=[AGENTS[SCHEDULING]],
                    gamma=PPO_CONF['discount_factor'],
                    lam=PPO_CONF['bias_variance_tradeoff'],
                    weight_policy_loss=PPO_CONF['policy_loss'],
                    weight_value_loss=PPO_CONF['value_loss'], 
                    weight_entropy_bonus=PPO_CONF['entropy'],
                    clipping_ratio=PPO_CONF['clip_ratio'])
            for e in range(epochs):
                print(f"\t Optimization epoch: {e+1}/{epochs}")
                scheduling_optimizer.zero_grad()
                training_loss, details = batch_result.compute_losses(agents, return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                scheduling_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _scheduling_loss_TRACKER.update(details.get(AGENTS[SCHEDULING]).policy_loss)
                losses.add(details)
            batch_results = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_value_loss_'+id)
    _scheduling_loss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_scheduling_loss_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_cmax_'+id)
    return agents, embedding_stack, shared_critic

def outsourcing_stage(target_instance : Instance, agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        Second (2nd) optimization stage: outsourcing agent alone
    """
    iterations: int = PPO_CONF['train_iterations']['fine_tuning'][0 if debug_mode else 1]
    epochs: int = PPO_CONF['opt_epochs']
    outsourcing_optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()), 
        lr=LEARNING_RATE)
    embedding_stack.train()
    shared_critic.train()
    _agent, _  = agents[OUTSOURCING]
    _agent.train()
    losses = MAPPO_Losses(agent_names=[AGENTS[OUTSOURCING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="pink", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _cost_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Cost", title="Final Cost by episode", color="orange", show=interactive)
    batch_results: list[MultiAgent_OneInstance] = []
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        loss, _, cmax, cost = solve_function(instance=target_instance, agents=agents, train=True, trainable=[1,0,0], device=device, debug_mode=debug_mode)
        _Cmax_TRACKER.update(cmax)
        _cost_TRACKER.update(cost)
        batch_results.append(loss)
        if episode % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=batch_results,
                    agent_names=[AGENTS[OUTSOURCING]],
                    gamma=PPO_CONF['discount_factor'],
                    lam=PPO_CONF['bias_variance_tradeoff'],
                    weight_policy_loss=PPO_CONF['policy_loss'],
                    weight_value_loss=PPO_CONF['value_loss'], 
                    weight_entropy_bonus=PPO_CONF['entropy'],
                    clipping_ratio=PPO_CONF['clip_ratio'])
            for e in range(epochs):
                print(f"\t Optimization epoch: {e+1}/{epochs}")
                outsourcing_optimizer.zero_grad()
                training_loss, details = batch_result.compute_losses(agents, return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                outsourcing_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _outsourcing_loss_TRACKER.update(details.get(AGENTS[OUTSOURCING]).policy_loss)
                losses.add(details)
            batch_results = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_value_loss_'+id)
    _outsourcing_loss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_outsourcing_loss_'+id)
    _cost_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_cost_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_cmax_'+id)
    return agents, embedding_stack, shared_critic

def multi_agent_stage(target_instance : Instance, agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, start_time: int, interactive: bool = False, debug_mode: bool=False):
    """
        Last (3rd) optimization stage: all three agents together
    """
    multi_agent_optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[SCHEDULING][AGENT].parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), 
        lr=LEARNING_RATE)
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()
    losses = MAPPO_Losses(agent_names=[name for _,name in agents])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="pink", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _cost_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Cost", title="Final Cost by episode", color="orange", show=interactive)
    _best_obj: float = math.inf
    _best_episode: int = 0
    _time_to_best: float = 0
    batch_results: list[MultiAgent_OneInstance] = []
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        loss, graph, cmax, cost = solve_function(instance=target_instance, agents=agents, train=True, trainable=[0,1,0], device=device, debug_mode=debug_mode)
        batch_results.append(loss)
        _current_obj = objective_value(cmax, cost, target_instance.w_makespan)
        if _current_obj < _best_obj:
            _best_obj = _current_obj
            _best_episode = episode
            _time_to_best = systime.time()-start_time
        _Cmax_TRACKER.update(cmax)
        _cost_TRACKER.update(cost)
        if episode % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=batch_results,
                    agent_names=[name for _,name in agents],
                    gamma=PPO_CONF['discount_factor'],
                    lam=PPO_CONF['bias_variance_tradeoff'],
                    weight_policy_loss=PPO_CONF['policy_loss'],
                    weight_value_loss=PPO_CONF['value_loss'], 
                    weight_entropy_bonus=PPO_CONF['entropy'],
                    clipping_ratio=PPO_CONF['clip_ratio'])
            for e in range(epochs):
                print(f"\t Optimization epoch: {e+1}/{epochs}")
                multi_agent_optimizer.zero_grad()
                training_loss, details = batch_result.compute_losses(agents, return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                multi_agent_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _scheduling_loss_TRACKER.update(details.get(AGENTS[SCHEDULING]).policy_loss)
                _outsourcing_loss_TRACKER.update(details.get(AGENTS[OUTSOURCING]).policy_loss)
                losses.add(details)
            batch_results = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/final_value_loss_'+id)
    _scheduling_loss_TRACKER.save(path+directory.solutions+'/'+size+'/final_scheduling_loss_'+id)
    _outsourcing_loss_TRACKER.save(path+directory.solutions+'/'+size+'/final_outsourcing_loss_'+id)
    _cost_TRACKER.save(path+directory.solutions+'/'+size+'/final_cost_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/final_cmax_'+id)
    with open(path+directory.solutions+'/'+size+'/fine_tune_gns_graph_'+id+'.pkl', 'wb') as f:
        pickle.dump(graph, f)
    with open(path+directory.solutions+'/'+size+'/fine_tune_gns_solution_'+id+'.pkl', 'wb') as f:
        pickle.dump(translate_solution(graph, target_instance), f)
    with open(path+directory.solutions+'/'+size+'/fine_tune_losses_'+id+'.pkl', 'wb') as f:
        pickle.dump(losses, f)
    return _best_obj, _best_episode, _time_to_best

def go_to_eval(agents: list[(Module, str)]):
    """
        Switch agents to eval mode
    """
    for agent,_ in agents:
        agent.eval()
    return agents

def multi_stage_fine_tuning(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        Multi-stage PPO function to fine-tune agents on the target instance
    """
    _itrs: int = PPO_CONF['train_iterations'][0 if debug_mode else 1]
    _epochs: int = PPO_CONF['opt_epochs']
    _batch_size: int = PPO_CONF['batch']
    _instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    print(f"Target instance {size}_{id} loaded....")
    _start_time = systime.time()
    go_to_eval(agents)
    print("I. FINE TUNING STAGE 1: scheduling agent")
    agents, embedding_stack, shared_critic = scheduling_stage(_instance, agents, embedding_stack, shared_critic, solve_function, path, _epochs, _itrs, _batch_size, device, id, size, interactive, debug_mode)
    go_to_eval(agents)
    print("II. FINE TUNING STAGE 2: outsourcing agent")
    agents, embedding_stack, shared_critic = outsourcing_stage(_instance, agents, embedding_stack, shared_critic, solve_function, path, _epochs, _itrs, _batch_size, device, id, size, interactive, debug_mode)
    print("III. FINE TUNING STAGE 3: multi-agent")
    _best_obj, _best_episode, _time_to_best = multi_agent_stage(_instance, agents, embedding_stack, shared_critic, solve_function, path, _epochs, _itrs, _batch_size, device, id, size, _start_time, interactive, debug_mode)
    final_metrics = pd.DataFrame({
        'index': [_instance.id],
        'value': [_best_obj],
        'episode': [_best_episode],
        'time_to_best': [_time_to_best],
        'computing_time': [systime.time()-_start_time],
        'device_used': [device]})
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_fine_tuned_gns_'+id+'.csv', index=False)