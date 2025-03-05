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
from model.agent import MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses
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

def multi_stage_fine_tuning(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, device: str, id: str, size: str, interactive: bool = False, debug_mode: bool=False):
    """
        PPO function to fine-tune agents on the target instance
    """
    optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[SCHEDULING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), 
        lr=LEARNING_RATE)
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
    
    