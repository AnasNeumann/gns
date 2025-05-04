from model.graph import State
import torch
from torch import Tensor
from typing import Tuple
import numpy as np
from torch.nn import Module
from tools.tensors import add_into_tensor
import torch.nn.functional as F

# ===========================================================
# =*= DATA MODEL FOR PPO AGENT CONFIGURATION AND RESULTS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

# START: VALIDATION ONLY ---------------------------------------------------------------------------
class Agent_Loss:
    def __init__(self, name: str):
        self.name = name
        self.policy_loss: float = 0.0
        self.entropy_bonus: float = 0.0

class MAPPO_Loss:
    def __init__(self, agent_names: list[str]):
        self.value_loss: float = 0.0
        self.agents: list[Agent_Loss] = [Agent_Loss(name) for name in agent_names]

    def get(self, name) -> Agent_Loss:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

class Agent_Losses:
    def __init__(self, name: str):
        self.name = name
        self.policy_loss: list[float] = []
        self.entropy_bonus: list[float] = []

class MAPPO_Losses:
    def __init__(self, agent_names: list[str]):
        self.value_loss: list[float] = []
        self.agents: list[Agent_Losses] = [Agent_Losses(name) for name in agent_names]

    def add(self, losses: MAPPO_Loss):
        self.value_loss.append(losses.value_loss)
        for agent in self.agents:
            corresponding_agent = losses.get(agent.name)
            agent.policy_loss.append(corresponding_agent.policy_loss)
            agent.entropy_bonus.append(corresponding_agent.entropy_bonus)

# END: VALIDATION ONLY ---------------------------------------------------------------------------

# START: TRANING ONLY ----------------------------------------------------------------------------
class Value_OneInstance:
    def __init__(self, agent_names: list[str], device: str):
        self.device = device
        self.gamma = 1
        self.cumulative_returns: Tensor = None
        self.values: Tensor = None
        self.agent_steps = {}
        self.reverse_agent_steps = []
        for name in agent_names:
            self.agent_steps[name] = []

    def add_step(self, value: Tensor, agent_name: str):
        agent_step = len(self.agent_steps[agent_name])
        global_step = 0 if self.values is None else len(self.values)
        self.agent_steps[agent_name].append(global_step)
        self.reverse_agent_steps.append((agent_name, agent_step))
        self.values = add_into_tensor(self.values, value)

    # R_t [sum version] = reward_t + gamma^1 * reward_(t+1) + ... + gamma^(T-t) * reward_T
    # R_t [recusive] = reward_t + gamma(R_(t+1))
    def compute_cumulative_returns(self, agents: dict):
        R = 0
        T = len(self.values)
        self.cumulative_returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            agent_name, agent_step = self.reverse_agent_steps[t]
            agent: Agent_OneInstance = agents[agent_name]
            R = agent.rewards[agent_step] + self.gamma * R
            self.cumulative_returns[t] = R

    # Value Loss (MSE version) = E_t[(values_t - cumulative_returns_t)^2]
    # Mean over all steps so value has a similar importance than agents (regardless of their use rate)
    # Huber loss in new version: L(pred values 'v', cumulative returns 'r') = 1/2 (r-v)^2 for small errors (|r-v| ≤ δ) else δ|r-v| - 1/2 x δ^2
    def compute_value_loss(self) -> Tensor:
        return F.smooth_l1_loss(self.values, self.cumulative_returns)

class Agent_OneInstance:
    def __init__(self, name: str, related_items: Tensor, parent_items: Tensor, w_makespan: Tensor, device: str):
        self.name = name
        self.device = device
        self.gamma = 1
        self.lam = 1
        self.clipping_ratio = 1
        self.related_items = related_items
        self.parent_items = parent_items
        self.w_makespan = w_makespan
        self.states: list[State] = []
        self.probabilities: list[Tensor]= []
        self.possibles_actions: list[(int, int)] = []
        self.actions_idx: list[int] = []
        self.rewards: Tensor = None
        self.advantages: Tensor = None

    def add_step(self, state: State, probabilities: Tensor, actions: Tuple[int, int], id: int):
        self.probabilities.append(probabilities)
        self.actions_idx.append(id)
        self.states.append(state)
        self.possibles_actions.append(actions)
    
    def add_reward(self, reward: float):
        r = torch.tensor([reward], device=self.device)
        self.rewards = add_into_tensor(self.rewards, r)
    
    # delta_t = reward_t + gamma*value_(t+1) - value_t
    # In this formula, t is step of the agent and global_t is the global step considering all 3 agents!
    def temporal_difference_residual(self, t: int, all_values: Value_OneInstance) -> Tensor:
        global_t = all_values.agent_steps[self.name][t]
        delta: Tensor = self.rewards[t] - all_values.values[global_t]
        if t >= len(self.states) - 1:
            return delta
        return delta + (self.gamma * all_values.values[global_t + 1])

    # GAE_t [sum version] = delta_t + (lam*gamma)^1 * delta_(t+1) + ... + (lam*gamma)^T-t+1 * delta_T
    # GAE_t [recursive] = delta_t + (lam*gamma) * GAE_(t+1)
    # --> Settings with values independant from agent!
    def compute_generalized_advantage_estimates(self, all_values: Value_OneInstance):
        GAE = 0
        T = len(self.states)
        self.advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            delta = self.temporal_difference_residual(t, all_values)
            GAE = delta + self.gamma * self.lam * GAE
            self.advantages[t] = GAE

    # Returns only policy loss and entropy bonus
    # ---------------------------------------------------------------------------------
    # Entropy bonus = E_t[-1 * SUM_a[probabilities(a|s_t) * LOG(probabilities(a|s_t))]] --> all probabilities!
    # ---------------------------------------------------------------------------------
    # probability_ratio_t [basic version] = new_probabilities(a_t|s_t) / old_probabilities(a_t|s_t)  --> only for selected actions!
    # probability_ratio_t [exp & log version] = e^[log_new_probabilities(a_t|s_t) - log_old_probabilities(a_t|s_t)]
    # Policy Loss = E_t[min(probability_ratio_t, CLIP[1-e, probability_ratio_t, 1+e]) * GAE_t]
    # ---------------------------------------------------------------------------------
    # MEAN over steps so agents have similar loss regarless of their use rate by instance
    def compute_PPO_losses(self, agent: Module):
        e = self.clipping_ratio
        log_new_probs: Tensor = None
        log_old_probs: Tensor = None
        entropies: Tensor = None
        for step, state in enumerate(self.states):
            new_probabilities,_ = agent(state, self.possibles_actions[step], self.related_items, self.parent_items, self.w_makespan)
            old_action_id: int = self.actions_idx[step]
            entropy = torch.sum(-new_probabilities*torch.log(new_probabilities+1e-8), dim=-1)
            new_log = torch.log(new_probabilities[old_action_id]+1e-8)
            old_log = torch.log(self.probabilities[step][old_action_id]+1e-8)
            if entropies is None:
                log_new_probs = new_log
                log_old_probs = old_log
                entropies = entropy
                log_new_probs = log_new_probs.to(self.device)
                log_old_probs = log_old_probs.to(self.device)
                entropies = entropies.to(self.device)
            else:
                entropies = torch.cat((entropies, entropy), dim=0)
                log_new_probs = torch.cat((log_new_probs, new_log), dim=0)
                log_old_probs = torch.cat((log_old_probs, old_log), dim=0)
        ratio: Tensor = torch.exp(log_new_probs-log_old_probs)
        policy_loss: Tensor = torch.min(ratio * self.advantages, torch.clamp(ratio, 1-e, 1+e) * self.advantages).mean()
        entropy_bonus: Tensor = torch.mean(entropies)
        policy_loss = policy_loss.to(self.device)
        entropy_bonus = entropy_bonus.to(self.device)
        return policy_loss, entropy_bonus

class MultiAgent_OneInstance:
    def __init__(self, agent_names: list[str], instance_id: int, related_items: Tensor, parent_items: Tensor, w_makespan: Tensor, device: str):
        self.instance_id = instance_id
        self.agents = {}
        self.value = Value_OneInstance(agent_names, device)
        for name in agent_names:
            self.agents[name] = Agent_OneInstance(name, related_items, parent_items, w_makespan, device)
    
    def add_step(self, agent_name: str, state: State, probabilities: Tensor, actions: Tuple[int, int], id: int, value: Tensor):
        agent: Agent_OneInstance = self.agents[agent_name]
        agent.add_step(state, probabilities, actions, id)
        self.value.add_step(value, agent_name)
    
    def add_reward(self, agent_name: str, reward: any):
        agent: Agent_OneInstance = self.agents[agent_name]
        agent.add_reward(reward)

class Value_Batch:
    def __init__(self, weight_value_loss: float):
        self.weight_value_loss: float = weight_value_loss
        self.instances: list[Value_OneInstance] = []

    # Value loss over batch (max version) = - w2*SUM_i[value_loss]
    # Value loss over batch (min version) = w2*SUM_i[value_loss]
    def compute_value_loss_over_batch(self, value_losses: Tensor):
        total_value_loss: Tensor = torch.mean(value_losses)
        print(f"\t\t value loss (over batch): {total_value_loss}")
        print("\t\t -----------------")
        return self.weight_value_loss*total_value_loss

class Agent_Batch:
    def __init__(self, name: str):
        self.name = name
        self.weight_policy_loss: float = 0.01
        self.weight_entropy_bonus: float = 0.01
        self.instances: list[Agent_OneInstance] = []

    # Loss for one agent over batch (max version) = w1*SUM_i[policy_loss] + w3*SUM_i[entropy_bonus]
    # Loss for one agent over batch (min version) = -w1*SUM_i[policy_loss] - w3*SUM_i[entropy_bonus]
    def compute_PPO_loss_over_batch(self, policy_losses: Tensor, entropy_bonuses: Tensor):
        total_policy_loss: Tensor = torch.mean(policy_losses)
        total_entropy_bonus: Tensor = torch.mean(entropy_bonuses)
        print(f"\t\t Computing losses for agent (over batch): {self.name}") 
        print(f"\t\t policy loss: {total_policy_loss}") 
        print(f"\t\t entropy bonus: {total_entropy_bonus}")
        print("\t\t -----------------")
        total_loss = -self.weight_policy_loss*total_policy_loss - self.weight_entropy_bonus*total_entropy_bonus
        return total_loss, total_policy_loss, total_entropy_bonus
        
class MultiAgents_Batch:
    def standardize_return_over_batch(self):
        all_returns = torch.cat([instance.cumulative_returns for instance in self.value_results.instances])
        mu_r, std_r = all_returns.mean(), all_returns.std().clamp_min(1e-8)
        for instance in self.value_results.instances: 
            instance.cumulative_returns = (instance.cumulative_returns - mu_r) / std_r
    
    def standardize_advantage_over_batch(self, agent: Agent_Batch):
        all_adv = torch.cat([instance.advantages for instance in agent.instances])
        mu_a, std_a = all_adv.mean(), all_adv.std().clamp_min(1e-8)
        for instance in agent.instances:
            instance.advantages = (instance.advantages - mu_a) / std_a
        return agent

    # Init alreay compile interdiate metrics: "cumulative returns" by instance (for value losses) and "generalized advantage estimates" of each agent by instance (for policy loss)
    def __init__(self, batch: list[MultiAgent_OneInstance], agent_names: list[str], gamma: float, lam: float, weight_policy_loss: float, weight_value_loss: float, weight_entropy_bonus: float, clipping_ratio: float):
        self.agents_results: list[Agent_Batch] = []
        self.value_results: Value_Batch = Value_Batch(weight_value_loss)
        for instance in batch:
            instance.value.gamma = gamma
            instance.value.compute_cumulative_returns(agents=instance.agents)
            self.value_results.instances.append(instance.value)
        batch = self.standardize_return_over_batch()
        for agent_name in agent_names:
            agent = Agent_Batch(name=agent_name)
            agent.weight_policy_loss = weight_policy_loss
            agent.weight_entropy_bonus = weight_entropy_bonus
            for instance in batch:
                agent_one_instance: Agent_OneInstance = instance.agents[agent_name]
                agent_one_instance.gamma = gamma
                agent_one_instance.lam = lam
                agent_one_instance.clipping_ratio = clipping_ratio
                agent_one_instance.compute_generalized_advantage_estimates(all_values=instance.value)
                agent.instances.append(agent_one_instance)
            agent = self.standardize_advantage_over_batch(agent)
            self.agents_results.append(agent)

    # Formula for multi agents L = scheduling_losses_over_batch (w1*policy + w2*value + w3*entropy) + outsourcing_losses_over_batch + material_losses_over_batch
    # SUM also means more importance for agents called more often (larger loss)!
    def compute_losses(self, agents: list[(Module, str)], return_details: bool) -> Tensor:
        losses: Tensor = None
        if return_details:
            details: MAPPO_Loss = MAPPO_Loss([name for _,name in agents])
        
        # 1/3 Compute value loss over complete batch
        value_losses: Tensor = None
        for value_instance in self.value_results.instances:
            value_loss_i = value_instance.compute_value_loss()
            value_losses = add_into_tensor(value_losses, value_loss_i.unsqueeze(0))
        v = self.value_results.compute_value_loss_over_batch(value_losses)
        losses = add_into_tensor(losses, v.unsqueeze(0))
        if return_details:
            details.value_loss = v.item()

        # 2/3 Compute policy loss over complete batch and combining all agents
        for agent, agent_name in agents:
            for results_of_one_agent in self.agents_results:
                if results_of_one_agent.name == agent_name:
                    policy_losses: Tensor = None
                    entropy_bonuses: Tensor = None
                    has_states_in_batch = False
                    for agent_instance in results_of_one_agent.instances:
                        if agent_instance.states:
                            has_states_in_batch = True
                            policy_loss_i, entropy_i, = agent_instance.compute_PPO_losses(agent)
                            policy_losses = add_into_tensor(policy_losses, policy_loss_i.unsqueeze(0))
                            entropy_bonuses = add_into_tensor(entropy_bonuses, entropy_i.unsqueeze(0))
                    if has_states_in_batch:
                        total_loss, p, e = results_of_one_agent.compute_PPO_loss_over_batch(policy_losses, entropy_bonuses)
                        losses = add_into_tensor(losses, total_loss.unsqueeze(0))
                        if return_details:
                            details.get(agent_name).policy_loss = p.item()
                            details.get(agent_name).entropy_bonus = e.item()
        
        # 3/3 Compute complete loss over batch
        if return_details:
            return torch.sum(losses), details
        else:
            return torch.sum(losses)