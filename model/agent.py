from model.graph import State
import torch
from torch import Tensor
from typing import Tuple
import numpy as np
from torch.nn import Module
import copy

# ===========================================================
# =*= DATA MODEL FOR PPO AGENT CONFIGURATION AND RESULTS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

CUDA = "cude"

class Agent_OneInstance:
    def __init__(self, name: str, related_items: Tensor, parent_items: Tensor, w_makespan: float, device: str):
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
        self.values: Tensor = None
        self.rewards: Tensor = None
        self.cumulative_returns: Tensor = None
        self.advantages: Tensor = None
        if device == CUDA:
            self.related_items.to(CUDA)
            self.parent_items.to(CUDA)

    def add_step(self, state: State, probabilities: Tensor, actions: Tuple[int, int], id: int, value: Tensor):
        if self.values is None:
            self.values = value
            self.values.to(self.device)
        else:
            self.values = torch.cat((self.values, value), dim=0)
        self.probabilities.append(probabilities)
        self.actions_idx.append(id)
        self.states.append(state)
        self.possibles_actions.append(actions)
    
    def add_reward(self, reward: float):
        if self.rewards is None:
            self.rewards = torch.Tensor([reward])
            self.rewards.to(self.device)
        else:
            self.rewards = torch.cat((self.rewards, torch.Tensor([reward])), dim=0)

    # R_t [sum version] = reward_t + gamma^1 * reward_(t+1) + ... + gamma^(T-t) * reward_T
    # R_t [recusive] = reward_t + gamma(R_(t+1))
    def compute_cumulative_returns(self):
        R = 0
        T = len(self.states)
        returns = np.zeros(T)
        for t in reversed(range(T)):
            R = self.rewards[t] + self.gamma * R
            returns[t] = R
        self.cumulative_returns = torch.tensor(returns, dtype=torch.float32)
        if self.device == CUDA:
           self.cumulative_returns.to(CUDA)

    # Value Loss = E_t[(values_t - cumulative_returns_t)^2]
    def compute_value_loss(self) -> Tensor:
        return torch.mean(torch.stack([(value - cumulative_return) ** 2 for value, cumulative_return in zip(self.values, self.cumulative_returns)]))
    
    # delta_t = reward_t + gamma*value_(t+1) - value_t
    def temporal_difference_residual(self, t) -> Tensor:
        delta: Tensor = self.rewards[t] - self.values[t]
        if t >= len(self.states) - 1:
            return delta
        return delta + (self.gamma * self.values[t+1])

    # GAE_t [sum version] = delta_t + (lam*gamma)^1 * delta_(t+1) + ... + (lam*gamma)^T-t+1 * delta_T
    # GAE_t [recursive] = delta_t + (lam*gamma) * GAE_(t+1)
    def compute_generalized_advantage_estimates(self):
        GAE = 0
        T = len(self.states)
        advantages = np.zeros(T)
        for t in reversed(range(T)):
            delta = self.temporal_difference_residual(t)
            GAE = delta + self.gamma * self.lam * GAE
            advantages[t] = GAE
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        if self.device == CUDA:
            self.advantages.to(CUDA)

    # Entropy bonus = E_t[-1 * SUM_a[probabilities(a|s_t) * LOG(probabilities(a|s_t))]] --> all probabilities!
    # ---------------------------------------------------------------------------------
    # probability_ratio_t [basic version] = new_probabilities(a_t|s_t) / old_probabilities(a_t|s_t)  --> only for selected actions!
    # probability_ratio_t [exp & log version] = e^[log_new_probabilities(a_t|s_t) - log_old_probabilities(a_t|s_t)]
    # Policy Loss = E_t[min(probability_ratio_t, CLIP[1-e, probability_ratio_t, 1+e]) * GAE_t]
    def compute_PPO_losses(self, agent: Module):
        e = self.clipping_ratio
        log_new_probs: Tensor = None
        log_old_probs: Tensor = None
        entropies: Tensor = None
        for step, state in enumerate(self.states):
            new_probabilities,_ = agent(copy.deepcopy(state), self.possibles_actions[step], self.related_items, self.parent_items, self.w_makespan)
            old_action_id: int = self.actions_idx[step]
            entropy = torch.sum(-new_probabilities*torch.log(new_probabilities+1e-8), dim=-1)
            new_log = torch.log(new_probabilities[old_action_id]+1e-8)
            old_log = torch.log(self.probabilities[step][old_action_id]+1e-8)
            if entropies is None:
                log_new_probs = new_log
                log_old_probs = old_log
                entropies = entropy
                log_new_probs.to(self.device)
                log_old_probs.to(self.device)
                entropies.to(self.device)
            else:
                entropies = torch.cat((entropies, entropy), dim=0)
                log_new_probs = torch.cat((log_new_probs, new_log), dim=0)
                log_old_probs = torch.cat((log_old_probs, old_log), dim=0)
        ratio: Tensor = torch.exp(log_new_probs-log_old_probs)
        policy_loss: Tensor = torch.min(ratio * self.advantages, torch.clamp(ratio, 1-e, 1+e) * self.advantages).mean()
        entropy_bonus: Tensor = torch.mean(entropies)
        value_loss: Tensor = self.compute_value_loss()
        if self.device == CUDA:
            policy_loss.to(CUDA)
            entropy_bonus.to(CUDA)
            value_loss.to(CUDA)
        return policy_loss, value_loss, entropy_bonus

class MultiAgent_OneInstance:
    def __init__(self, agent_names: list[str], instance_id: int, related_items: Tensor, parent_items: Tensor, w_makespan: float, device: str):
        self.instance_id = instance_id
        self.agents: list[Agent_OneInstance] = []
        for name in agent_names:
            self.agents.append(Agent_OneInstance(name, related_items, parent_items, w_makespan, device))

    def get(self, name: str) -> Agent_OneInstance:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def add_step(self, agent_name: str, state: State, probabilities: Tensor, actions: Tuple[int, int], id: int, value: Tensor):
        agent = self.get(name=agent_name)
        if agent is not None:
            agent.add_step(state, probabilities, actions, id, value)
    
    def add_reward(self, agent_name: str, reward: any):
        agent = self.get(name=agent_name)
        if agent is not None:
            agent.add_reward(reward)

class Agent_Batch:
    def __init__(self, name: str):
        self.name = name
        self.weight_policy_loss: float = 0.01
        self.weight_value_loss: float = 0.01
        self.weight_entropy_bonus: float = 0.01
        self.instances: list[Agent_OneInstance] = []

    # Loss(max version) = w1*SUM_i[policy_loss] - w2*SUM_i[value_loss] + w3*SUM_i[entropy_bonus]
    # Loss(min version) = -w1*SUM_i[policy_loss] + w2*SUM_i[value_loss] - w3*SUM_i[entropy_bonus]
    def compute_PPO_loss_over_batch(self, policy_losses: list[Tensor], value_losses: list[Tensor], entropy_bonuses: list[Tensor]) -> Tensor:
        total_policy_loss: Tensor = sum(policy_losses)
        total_value_loss: Tensor = sum(value_losses)
        total_entropy_bonus: Tensor = sum(entropy_bonuses)
        print(f"\t\t Computing losses for agent: {self.name}") 
        print(f"\t\t policy loss: {total_policy_loss}") 
        print(f"\t\t value loss: {total_value_loss}")
        print(f"\t\t entropy bonus: {total_entropy_bonus}")
        print("\t\t -----------------")
        return -self.weight_policy_loss*total_policy_loss + self.weight_entropy_bonus*total_value_loss - self.weight_entropy_bonus*total_entropy_bonus
    
class MultiAgents_Batch:
    def __init__(self, batch: list[MultiAgent_OneInstance], agent_names: list[str], gamma: float, lam: float, weight_policy_loss: float, weight_value_loss: float, weight_entropy_bonus: float, clipping_ratio: float):
        self.agents_results: list[Agent_Batch] = []
        for agent_name in agent_names:
            agent = Agent_Batch(name=agent_name)
            agent.weight_policy_loss = weight_policy_loss
            agent.weight_value_loss = weight_value_loss
            agent.weight_entropy_bonus = weight_entropy_bonus
            for instance in batch:
                agent_of_instance: Agent_OneInstance = instance.get(name=agent_name)
                if agent_of_instance is not None:
                    agent_of_instance.gamma = gamma
                    agent_of_instance.lam = lam
                    agent_of_instance.clipping_ratio = clipping_ratio
                    agent_of_instance.compute_cumulative_returns()
                    agent_of_instance.compute_generalized_advantage_estimates()
                    agent.instances.append(agent_of_instance)
            self.agents_results.append(agent)

    def compute_losses(self, agents: list[(Module, str)]) -> Tensor:
        losses: list[Tensor] = []
        for agent, agent_name in agents:
            for results_of_one_agent in self.agents_results:
                if results_of_one_agent.name == agent_name:
                    policy_losses: list[Tensor] = []
                    value_losses: list[Tensor] = []
                    entropy_bonuses: list[Tensor] = []
                    has_states = False
                    for agent_instance in results_of_one_agent.instances:
                        if agent_instance.states:
                            has_states = True
                            p, v, e, = agent_instance.compute_PPO_losses(agent)
                            policy_losses.append(p)
                            value_losses.append(v)
                            entropy_bonuses.append(e)
                    if has_states:
                        losses.append(results_of_one_agent.compute_PPO_loss_over_batch(policy_losses, value_losses, entropy_bonuses))
        return sum(losses)