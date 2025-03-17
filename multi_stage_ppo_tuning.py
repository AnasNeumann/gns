import pickle
import os
from model.instance import Instance
from tools.common import directory
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from torch.nn import Module
from torch.optim import Optimizer
from typing import Callable
from model.agent import MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses, MultiAgent_OneInstance
import time as systime
from tools.common import load_instance, objective_value, freeze, unfreeze, unfreeze_all, freeze_several_and_unfreeze_others
from translators.graph2solution_translator import translate_solution
from debug.loss_tracker import LossTracker
import math
from model.reward_memory import Memory

# ==================================================================
# =*= MULTI-STAGE PROXIMAL POLICY OPTIMIZATION (PPO) FINE TUNING =*=
# ==================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PPO_CONF = {
    "train_iterations": [300, 150, 150, 150, 300],
    "opt_epochs": 3,
    "clip_ratio": 0.1,
    "policy_loss": 1.0,
    "batch": 16,
    "value_loss": 0.5,
    "entropy": 0.02,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0
}
LEARNING_RATE = 1e-3
AGENTS = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, run_number:int, complete_path: str):
    index = str(run_number)
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights_'+index+'.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights_'+index+'.pth')
    torch.save(optimizer.state_dict(), complete_path+'/adam_'+index+'.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights_'+index+'.pth')

def scheduling_stage(target_instance : Instance, agents: list[(Module, str)], solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, interactive: bool = False):
    """
        First (1st) optimization stage: scheduling agent alone
    """
    
    scheduling_optimizer = torch.optim.Adam(list(agents[SCHEDULING][AGENT].parameters()), lr=LEARNING_RATE)
    losses = MAPPO_Losses(agent_names=[AGENTS[SCHEDULING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _batch_replay_memory: list[MultiAgent_OneInstance] = []
    reward_MEMORY: Memory = Memory(target_instance.id)
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        transitions, reward_MEMORY, _, cmax, _ = solve_function(instance=target_instance, agents=agents, train=True, trainable=[0,1,0], device=device, fixed_alpha=1.0, reward_MEMORY=reward_MEMORY)
        _batch_replay_memory.append(transitions)
        _Cmax_TRACKER.update(cmax)
        if (episode+1) % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=_batch_replay_memory,
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
                training_loss, details = batch_result.compute_losses([agents[SCHEDULING]], return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                scheduling_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _scheduling_loss_TRACKER.update(details.get(AGENTS[SCHEDULING]).policy_loss)
                losses.add(details)
            _batch_replay_memory = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_value_loss_'+id)
    _scheduling_loss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_scheduling_loss_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/stage_1_cmax_'+id)
    return agents
    
def material_use_stage(target_instance : Instance, agents: list[(Module, str)], solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, interactive: bool = False):
    """
        Second (2nd) optimization stage: material use agent alone
    """

    material_optimizer = torch.optim.Adam(list(agents[MATERIAL_USE][AGENT].parameters()),  lr=LEARNING_RATE)
    losses = MAPPO_Losses(agent_names=[AGENTS[MATERIAL_USE]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _material_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Material use loss (policy)", title="Material use loss (policy)", color="green", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _batch_replay_memory: list[MultiAgent_OneInstance] = []
    reward_MEMORY: Memory = Memory(target_instance.id)
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        transitions, reward_MEMORY, _, cmax, _ = solve_function(instance=target_instance, agents=agents, train=True, trainable=[0,0,1], device=device, fixed_alpha=1.0, reward_MEMORY=reward_MEMORY)
        _batch_replay_memory.append(transitions)
        _Cmax_TRACKER.update(cmax)
        if (episode+1) % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=_batch_replay_memory,
                    agent_names=[AGENTS[MATERIAL_USE]],
                    gamma=PPO_CONF['discount_factor'],
                    lam=PPO_CONF['bias_variance_tradeoff'],
                    weight_policy_loss=PPO_CONF['policy_loss'],
                    weight_value_loss=PPO_CONF['value_loss'], 
                    weight_entropy_bonus=PPO_CONF['entropy'],
                    clipping_ratio=PPO_CONF['clip_ratio'])
            for e in range(epochs):
                print(f"\t Optimization epoch: {e+1}/{epochs}")
                material_optimizer.zero_grad()
                training_loss, details = batch_result.compute_losses([agents[MATERIAL_USE]], return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                material_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _material_loss_TRACKER.update(details.get(AGENTS[MATERIAL_USE]).policy_loss)
                losses.add(details)
            _batch_replay_memory = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_value_loss_'+id)
    _material_loss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_material_use_loss_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/stage_2_cmax_'+id)
    return agents

def outsourcing_stage(target_instance : Instance, agents: list[(Module, str)], solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, substage: int, interactive: bool = False):
    """
        Third (3rd) optimization stage: outsourcing agent alone
    """
    outsourcing_optimizer = torch.optim.Adam(list(agents[OUTSOURCING][AGENT].parameters()), lr=LEARNING_RATE)
    losses = MAPPO_Losses(agent_names=[AGENTS[OUTSOURCING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="pink", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _cost_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Cost", title="Final Cost by episode", color="orange", show=interactive)
    _batch_replay_memory: list[MultiAgent_OneInstance] = []
    reward_MEMORY: Memory = Memory(target_instance.id)
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        transitions, reward_MEMORY, _, cmax, cost = solve_function(instance=target_instance, agents=agents, train=True, trainable=[1,0,0], device=device, reward_MEMORY=reward_MEMORY)
        _Cmax_TRACKER.update(cmax)
        _cost_TRACKER.update(cost)
        _batch_replay_memory.append(transitions)
        if (episode+1) % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=_batch_replay_memory,
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
                training_loss, details = batch_result.compute_losses([agents[OUTSOURCING]], return_details=True)
                print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
                training_loss.backward(retain_graph=False)
                outsourcing_optimizer.step()
                details: MAPPO_Loss
                _vloss_TRACKER.update(details.value_loss)
                _outsourcing_loss_TRACKER.update(details.get(AGENTS[OUTSOURCING]).policy_loss)
                losses.add(details)
            _batch_replay_memory = []
    _vloss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_3_'+str(substage)+'_value_loss_'+id)
    _outsourcing_loss_TRACKER.save(path+directory.solutions+'/'+size+'/stage_3_'+str(substage)+'_outsourcing_loss_'+id)
    _cost_TRACKER.save(path+directory.solutions+'/'+size+'/stage_3_'+str(substage)+'_cost_'+id)
    _Cmax_TRACKER.save(path+directory.solutions+'/'+size+'/stage_3_'+str(substage)+'_cmax_'+id)
    return agents

def multi_agent_stage(target_instance : Instance, agents: list[(Module, str)], solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, device: str, id: str, size: str, start_time: int, interactive: bool = False):
    """
        Last (4th) optimization stage: all three agents together
    """
    multi_agent_optimizer = torch.optim.Adam(list(agents[SCHEDULING][AGENT].parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), lr=LEARNING_RATE)
    losses = MAPPO_Losses(agent_names=[name for _,name in agents])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs (3 per batch of episodes)", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="pink", show=interactive)
    _Cmax_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _cost_TRACKER: LossTracker = LossTracker(xlabel="Solving episode", ylabel="Cost", title="Final Cost by episode", color="orange", show=interactive)
    _best_obj: float = math.inf
    _best_episode: int = 0
    _time_to_best: float = 0
    _batch_replay_memory: list[MultiAgent_OneInstance] = []
    reward_MEMORY: Memory = Memory(target_instance.id)
    for episode in range(iterations):
        print(f"PPO solving episode: {episode+1}/{iterations}...")
        transitions, reward_MEMORY, graph, cmax, cost = solve_function(instance=target_instance, agents=agents, train=True, trainable=[1,1,1], device=device, reward_MEMORY=reward_MEMORY)
        _batch_replay_memory.append(transitions)
        _current_obj = objective_value(cmax, cost, target_instance.w_makespan)
        if _current_obj < _best_obj:
            _best_obj = _current_obj
            _best_episode = episode
            _time_to_best = systime.time()-start_time
        _Cmax_TRACKER.update(cmax)
        _cost_TRACKER.update(cost)
        if (episode+1) % batch_size == 0:
            print(f"PPO time for optimization after episode: {episode+1}/{iterations}!")
            batch_result: MultiAgents_Batch = MultiAgents_Batch(
                    batch=_batch_replay_memory,
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
            _batch_replay_memory = []
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

def multi_stage_fine_tuning(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        Multi-stage PPO function to fine-tune agents on the target instance
    """
    _itrs: int = PPO_CONF['train_iterations']
    _epochs: int = PPO_CONF['opt_epochs']
    _batch_size: int = PPO_CONF['batch']
    _instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    print(f"Target instance {size}_{id} loaded....")
    _start_time = systime.time()
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()
    
    print("I. FINE TUNING STAGE 1: scheduling agent")
    freeze_several_and_unfreeze_others(agents, [AGENTS[OUTSOURCING], AGENTS[MATERIAL_USE]])
    agents = scheduling_stage(_instance, agents, solve_function, path, _epochs, _itrs[0], _batch_size, device, id, size, interactive)
    
    print("II. FINE TUNING STAGE 2: material use agent with freezed embedding and critic layers")
    freeze_several_and_unfreeze_others(agents, [AGENTS[SCHEDULING], AGENTS[OUTSOURCING]])
    freeze(embedding_stack)
    freeze(shared_critic)
    agents = material_use_stage(_instance, agents, solve_function, path, _epochs, _itrs[1], _batch_size, device, id, size, interactive)
    
    print("III. FINE TUNING STAGE 3.1: outsourcing agent with freezed embedding and critic layers")
    freeze_several_and_unfreeze_others(agents, [AGENTS[SCHEDULING], AGENTS[MATERIAL_USE]])
    freeze(embedding_stack)
    freeze(shared_critic)
    agents = outsourcing_stage(_instance, agents, solve_function, path, _epochs, _itrs[2], _batch_size, device, id, size, 1, interactive)
    
    print("III. FINE TUNING STAGE 3.2: outsourcing agent and all layers")
    unfreeze(embedding_stack)
    unfreeze(shared_critic)
    agents = outsourcing_stage(_instance, agents, solve_function, path, _epochs, _itrs[3], _batch_size, device, id, size, 2, interactive)

    print("IV. FINE TUNING STAGE 4: multi-agent with all layers")
    unfreeze_all(agents)
    _best_obj, _best_episode, _time_to_best = multi_agent_stage(_instance, agents, solve_function, path, _epochs, _itrs[4], _batch_size, device, id, size, _start_time, interactive)
    final_metrics = pd.DataFrame({
        'index': [_instance.id],
        'value': [_best_obj],
        'episode': [_best_episode],
        'time_to_best': [_time_to_best],
        'computing_time': [systime.time()-_start_time],
        'device_used': [device]})
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_fine_tuned_gns_'+id+'.csv', index=False)