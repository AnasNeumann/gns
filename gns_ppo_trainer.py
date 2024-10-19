import pickle
import os
from model.instance import Instance
from model.graph import State
from model.gnn import L1_EmbbedingGNN, L1_CommonCritic
from common import directory
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
import copy
from torch.multiprocessing import Pool

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
LEARNING_RATE = 2e-4
PROBLEM_SIZES = [['s', 'm'], ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']]
PPO_CONF = {
    "validation_rate": 10,
    "switch_batch": 20,
    "train_iterations": [3, 1000], 
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

def save_models(agents: list[(Module, str)], embedding_stack: L1_EmbbedingGNN, shared_critic: L1_CommonCritic, path: str):
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

def simulate_solving_for_gradients(instance_id: int, related_items: Tensor, parent_items: Tensor, alpha: Tensor, agents: list[(Module, str)], states: list[State], agent_by_state: list[int], possible_actions_by_state: list[(int, int)], action_id_by_state: list[int], reward_by_state: list[float], device: str):
    training_results: MultiAgent_OneInstance = MultiAgent_OneInstance(
            agent_names=[name for _,name in agents], 
            instance_id=instance_id,
            related_items=related_items,
            parent_items=parent_items,
            w_makespan=alpha,
            device=device)
    for t, state in enumerate(states):
        temp_state = copy.deepcopy(state)
        temp_state.to(device=device)
        probs, state_value = agents[agent_by_state[t]][AGENT](temp_state, possible_actions_by_state[t], related_items, parent_items, alpha)
        training_results.add_step(
            agent_name=ACTIONS_NAMES[agent_by_state[t]], 
            state=state,
            probabilities=probs.detach(),
            actions=action_id_by_state[t],
            id=action_id_by_state[t],
            reward=reward_by_state[t],
            value=state_value.detach())
    return training_results

def models_to_train(agents: list[(Module, str)], embedding_stack: L1_EmbbedingGNN, shared_critic: L1_CommonCritic):
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()

def models_to_eval(agents: list[(Module, str)], embedding_stack: L1_EmbbedingGNN, shared_critic: L1_CommonCritic):
    embedding_stack.eval()
    shared_critic.eval()
    for agent,_ in agents:
        agent.eval()

def async_real_solve(init_args):
    solve_function, instance, agents, device, debug = init_args
    with torch.no_grad():
        result =  solve_function(instance, agents, "", True, device, debug)
    return result

def train_or_validate_batch(agents: list[(Module, str)], embedding_stack: L1_EmbbedingGNN, shared_critic: L1_CommonCritic, batch: list[Instance],train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str, num_processes: int, debug: bool):
    models_to_eval(agents, embedding_stack, shared_critic)
    print(f"\t Start the real solving in parallel (1/2)...")
    with Pool(num_processes) as pool:
        results = pool.map(async_real_solve, [(solve_function, instance, agents, device, debug) for instance in batch])
    all_instances_idx, all_instances_related_items, all_instances_parent_items, all_instances_alpha, all_instances_states, all_instances_agent_by_state, all_instances_possible_actions_by_state, all_instances_action_id_by_state, all_instances_reward_by_state = zip(*results)
    if train:
        models_to_train(agents, embedding_stack, shared_critic)
    instances_results: list[MultiAgent_OneInstance] = []
    for i,_ in enumerate(batch):
        print(f"\t Start simulating the solving of instance: {all_instances_idx[i]} in serial (2/2)...")
        instances_results.append(simulate_solving_for_gradients(instance_id=all_instances_idx[i], 
                                       related_items=all_instances_related_items[i], 
                                       parent_items=all_instances_parent_items[i], 
                                       alpha=all_instances_alpha[i],
                                       agents=agents, 
                                       states=all_instances_states[i], 
                                       agent_by_state=all_instances_agent_by_state[i], 
                                       possible_actions_by_state=all_instances_possible_actions_by_state[i], 
                                       action_id_by_state=all_instances_action_id_by_state[i], 
                                       reward_by_state=all_instances_reward_by_state[i], 
                                       device=device))
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

def PPO_train(agents: list[(Module, str)], embedding_stack: L1_EmbbedingGNN, shared_critic: L1_CommonCritic, path: str, solve_function: Callable, num_processes: int, debug_mode: bool=False):
    start_time = systime.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations: int = PPO_CONF['train_iterations'][0 if debug_mode else 1]
    batch_size: int = PPO_CONF['batch_size'][0 if debug_mode else 1]
    epochs: int = PPO_CONF['opt_epochs']
    debug_print: Callable = debug_printer(debug_mode)
    print("Loading dataset....")
    instances: list[Instance] = load_training_dataset(path=path, debug_mode=debug_mode)
    print(f"Dataset loaded after {(systime.time()-start_time)} seconds!")
    embedding_stack.to(device)
    shared_critic.to(device)
    vlosses = MAPPO_Losses(agent_names=[name for _,name in agents])
    for agent,_ in agents:
        agent.to(device)
    optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(embedding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[SCHEDULING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), 
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
        train_or_validate_batch(agents=agents, 
                                embedding_stack=embedding_stack, 
                                shared_critic=shared_critic, 
                                batch=current_batch, 
                                train=True, 
                                epochs=epochs, 
                                optimizer=optimizer, 
                                solve_function=solve_function, 
                                device=device, 
                                num_processes=num_processes,
                                debug=debug_mode)
        if iteration % PPO_CONF['validation_rate'] == 0:
            debug_print("\t time to validate the loss...")
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents=agents, 
                                                                    embedding_stack=embedding_stack, 
                                                                    shared_critic=shared_critic, 
                                                                    batch=val_instances, 
                                                                    train=False, 
                                                                    epochs=-1, 
                                                                    optimizer=None, 
                                                                    solve_function=solve_function, 
                                                                    device=device, 
                                                                    num_processes=num_processes,
                                                                    debug=debug_mode)
                vlosses.add(current_vloss)
    with open(directory.models+'/validation.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents, embedding_stack, shared_critic, path=path)
    print("<======***--| END OF TRAINING |--***======>")