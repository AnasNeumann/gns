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
    "batch_size": [3, 20],
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

def calculate_returns(rewards: list[int], gamma: float=PPO_CONF['discount_factor']):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def generalized_advantage_estimate(rewards: list[int], values: list[float], gamma: float=PPO_CONF['discount_factor'], lam: float=PPO_CONF['bias_variance_tradeoff']):
    GAE = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] - values[t]
        if t<len(rewards)-1:
            delta = delta + (gamma * values[t+1])
        GAE = delta + gamma * lam * GAE
        advantages.insert(0, GAE)
    return advantages

def PPO_loss(instances: list[Instance], agent: list, old_probs: Tensor, states: list[State], actions: list[list[(int, int)]], actions_idx: list[int], advantages: list[float], old_values: list[Tensor], returns: list[Tensor], instances_idx: list[int], all_related_items: list[generic_object], all_parents: list[generic_object], e: float=PPO_CONF['clip_ratio']):
    new_log_probs = torch.Tensor([])
    old_log_probs = torch.Tensor([])
    entropies = torch.Tensor([])
    id = -1
    instance = None
    related_items = []
    parents = []
    for i in range(len(states)):
        if instances_idx[i] != id:
            id = instances_idx[i]
            instance = search_instance(instances, id)
            related_items = search_object_by_id(all_related_items, id)['related_items']
            parents = search_object_by_id(all_parents, id)['parents']
        p,_ = agent(states[i], actions[i], related_items, parents, instance.w_makespan)
        a = actions_idx[i]
        entropies = torch.cat((entropies, torch.sum(-p*torch.log(p+1e-8), dim=-1)))
        new_log_probs = torch.cat((new_log_probs, torch.log(p[a]+1e-8)))
        old_log_probs = torch.cat((old_log_probs, torch.log(old_probs[i][a]+1e-8)))
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-e, 1+e)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = torch.mean(torch.stack([(V_old - r) ** 2 for V_old, r in zip(old_values, returns)]))
    entropy_loss = torch.mean(entropies)
    print(f"\t\t value loss - {value_loss}")
    print(f"\t\t policy loss - {policy_loss}") 
    print(f"\t\t entropy loss - {entropy_loss}") 
    return (PPO_CONF['policy_loss']*policy_loss) + (PPO_CONF['value_loss']*value_loss) - (entropy_loss*PPO_CONF['entropy'])

def PPO_optimize(optimizer: Optimizer, loss: float):
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
    all_probabilities = [[] for _ in agents]
    all_states = [[] for _ in agents]
    all_actions = [[] for _ in agents]
    all_actions_idx = [[] for _ in agents]
    all_instances_idx = [[] for _ in agents]
    all_related_items, all_parents = [], []
    with Pool(num_processes) as pool:
        results = pool.map(async_solve_one, [(agents, instance, debug, solve_function) for instance in batch])
    all_rewards, all_values, probabilities, states, actions, actions_idx, instances_idx, related_items, parents = zip(*results)
    for instance in range(len(batch)):
        all_parents.append({'id': instances_idx[instance][0][0], 'parents': parents[instance]})
        all_related_items.append({'id': instances_idx[instance][0][0], 'related_items': related_items[instance]}) 
        for agent_id in range(len(agents)):
            all_states[agent_id].extend(states[instance][agent_id])
            all_probabilities[agent_id].extend(probabilities[instance][agent_id])
            all_actions[agent_id].extend(actions[instance][agent_id])
            all_actions_idx[agent_id].extend(actions_idx[instance][agent_id])
            all_instances_idx[agent_id].extend(instances_idx[instance][agent_id])
    all_returns = [[ri for r in agent_rewards for ri in calculate_returns(r)] for agent_rewards in all_rewards]
    advantages = []
    flattened_values = []
    for agent_id, _ in enumerate(agents):
        advantages.append(torch.Tensor([gae for r, v in zip(all_rewards[agent_id], all_values[agent_id]) for gae in generalized_advantage_estimate(r, v)]))
        flattened_values.append([v for vals in all_values[agent_id] for v in vals])
    if train and epochs>0:
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            for agent_id, (agent, name) in enumerate(agents):
                print(f"\t\t Optimizing agent: {name}...")
                loss = PPO_loss(
                    instances=batch, 
                    agent=agent, 
                    old_probs=all_probabilities[agent_id], 
                    states=all_states[agent_id], 
                    actions=all_actions[agent_id], 
                    actions_idx=all_actions_idx[agent_id], 
                    advantages=advantages[agent_id], 
                    old_values=flattened_values[agent_id], 
                    returns=all_returns[agent_id], 
                    instances_idx=all_instances_idx[agent_id], 
                    all_related_items=all_related_items, 
                    all_parents=all_parents)
                PPO_optimize(optimizers[agent_id], loss)
    else:
        for agent_id, (agent, name) in enumerate(agents):
            print(f"\t\t Evaluating agent: {name}...")
            loss = PPO_loss(
                    instances=batch, 
                    agent=agent, 
                    old_probs=all_probabilities[agent_id], 
                    states=all_states[agent_id], 
                    actions=all_actions[agent_id], 
                    actions_idx=all_actions_idx[agent_id], 
                    advantages=advantages[agent_id], 
                    old_values=flattened_values[agent_id], 
                    returns=all_returns[agent_id], 
                    instances_idx=all_instances_idx[agent_id], 
                    all_related_items=all_related_items, 
                    all_parents=all_parents)
            print(f'\t Average Loss = {loss:.4f}')

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