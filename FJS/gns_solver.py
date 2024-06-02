import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool
from common import load_instances, OP_STRUCT, shape
from torch.nn import Sequential as Seq, Linear as Lin, ELU, Tanh, Parameter
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import copy
import pandas as pd 
import random
import numpy as np

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
SCHEDULED = 1
OP_FEATURES = {
    "status": 0, 
    "remaining_neighboring_resources": 1, 
    "duration": 2, 
    "start": 3, 
    "job_unscheduled_ops": 4, 
    "current_job_completion": 5
}
RES_FEATURES = {
    "available_time": 0, 
    "remaining_neighboring_ops": 1, 
    "past_utilization_rate": 2
}
GAT_CONF = {
    "gnn_layers": 2, 
    "embedding_dims": 8, 
    "MLP_size": 128, 
    "actor_critic_dim": 64
}
PPO_CONF = {
    "validation_rate": 10, 
    "switch_batch": 20, 
    "train_iterations": 1000, 
    "opt_epochs": 3,
    "batch_size": 5, # 20, 
    "clip_ratio": 0.2,
    "policy_loss": 1.0, 
    "value_loss": 0.5, 
    "entropy": 0.01, 
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation_ratio': 0.1
}
OPT_CONF = {"learning_rate": 2e-4}
SAMPLES = 100

#====================================================================================================================
# =*= I. FONCTIONS TO BUILD THE INITIAL GRAPH =*=
#====================================================================================================================

def init_start_time(job, position):
    return sum(o[OP_STRUCT["duration"]] for o in job[:position])

def init_end_time(job, position):
    return init_start_time(job, position) + job[position][OP_STRUCT["duration"]]

def init_ops_by_resource(type, jobs):
    return len([op for job in jobs for op in job if op[OP_STRUCT["resource_type"]]==type])

def add_node(graph, type, features):
    graph[type].x = torch.cat([graph[type].x, features], dim=0) if type in graph.node_types else features
    return graph

def add_edge(graph, first, middle, last, features):
    graph[first, middle, last].edge_index = torch.cat([graph[first, middle, last].edge_index, features], dim=1) if (first, middle, last) in graph.edge_types else features
    return graph

def instance_to_graph(instance):
    graph = HeteroData()
    resources = instance['resources']
    jobs = instance['jobs']
    zero_features = torch.zeros((1, 6), dtype=torch.int64)
    graph = add_node(graph, 'operation', zero_features)
    operations2graph = [] 
    resources_by_type = []
    graph2instance = []

    resource_id = 0
    for type, quantity in enumerate(resources):
        res_of_type = []
        for _ in range(quantity):
            graph = add_node(graph, 'resource', torch.tensor([[0, init_ops_by_resource(type, jobs), 0]], dtype=torch.float))
            res_of_type.append(resource_id)
            resource_id += 1
        resources_by_type.append(res_of_type)
    
    op_id = 1
    lower_makespan = 0
    for job_id, job in enumerate(jobs):
        first_op_id = op_id
        ops2graph = []
        end_job = init_end_time(job, len(job)-1)
        lower_makespan = max(lower_makespan, end_job)
        unscheduled_ops = len(job)
        for position, operation in enumerate(job):
            duration = operation[OP_STRUCT["duration"]]
            available_resources = resources[operation[OP_STRUCT["resource_type"]]]
            graph = add_node(graph, 'operation', torch.tensor([[NOT_SCHEDULED, available_resources, duration, init_start_time(job, position), unscheduled_ops, end_job]], dtype=torch.float))
            ops2graph.append(op_id)
            graph2instance.append((job_id, position))
            if op_id > first_op_id:
                graph = add_edge(graph, 'operation', 'precedence', 'operation', torch.tensor([[op_id - 1], [op_id]], dtype=torch.long))
            op_id += 1
        operations2graph.append(ops2graph)  

    graph = add_node(graph, 'operation', zero_features)
    first_op = 1
    for job_id, job in enumerate(jobs):
        graph = add_edge(graph, 'operation', 'precedence', 'operation', torch.tensor([[0], [first_op]], dtype=torch.long))
        first_op += len(job)
        graph = add_edge(graph, 'operation', 'precedence', 'operation', torch.tensor([[first_op-1], [instance['size']+1]], dtype=torch.long))
        for op_id, operation in enumerate(job):
            op2graph_id = operations2graph[job_id][op_id]
            for res2graph_id in resources_by_type[operation[OP_STRUCT["resource_type"]]]:
                graph = add_edge(graph, 'operation', 'uses', 'resource', torch.tensor([[op2graph_id], [res2graph_id]], dtype=torch.long))       
    return graph, graph2instance, lower_makespan

def display_graph(graph):
    print("Graph_overview: " + str(graph))
    print("Resources: " + str(graph['resource']))
    print("Operations: " + str(graph['operation']))
    print("Precedence_relations: " + str(graph['operation', 'precedence', 'operation']))
    print("Requirements operation->resource: " + str(graph['operation', 'uses', 'resource']))

#====================================================================================================================
# =*= II. GRAPH ATTENTION NEURAL NET ARCHITECTURE: TWO-STAGE EMBEDDING + ACTOR-CRITIC HEADS =*=
#====================================================================================================================

class ResourceAttentionEmbeddingLayer(MessagePassing):
    def __init__(self, resource_features_dim, operation_features_dim):
        super(ResourceAttentionEmbeddingLayer, self).__init__()
        self.hidden_dim = GAT_CONF["embedding_dims"]
        self.resource_transform = Lin(resource_features_dim, self.hidden_dim, bias=False)
        self.att_self_coef = Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        self.operation_transform = Lin(operation_features_dim, self.hidden_dim, bias=False)
        self.att_coef = Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.resource_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.operation_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_self_coef.data, gain=1.414)

    def forward(self, resources, operations, requirement_edges):
        resources = self.resource_transform(resources) 
        operations = self.operation_transform(operations)
        ops_by_edges = operations[requirement_edges[0]]
        res_by_edges = resources[requirement_edges[1]]
        self_attention = self.leaky_relu(torch.matmul(torch.cat([resources, resources], dim=-1), self.att_self_coef))
        cross_attention = self.leaky_relu(torch.matmul(torch.cat([res_by_edges, ops_by_edges], dim=-1), self.att_coef))
        normalizer = F.softmax(torch.cat([self_attention, cross_attention], dim=0), dim=0)
        norm_self_attention = normalizer[:self_attention.size(0)]
        norm_cross_attention = normalizer[self_attention.size(0):]
        weighted_ops_by_edges = norm_cross_attention * ops_by_edges
        sum_ops_by_edges = torch.zeros_like(resources)
        sum_ops_by_edges.index_add_(0, requirement_edges[1], weighted_ops_by_edges)
        embedding = F.elu(norm_self_attention * resources + sum_ops_by_edges)
        return embedding

class OperationEmbeddingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(OperationEmbeddingLayer, self).__init__()
        self.hidden_channels = GAT_CONF["MLP_size"]
        self.embedding_size = out_channels
        self.mlp_combined = Seq(
            Lin(4 * out_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, out_channels)
        )
        self.mlp_predecessor = Seq(
            Lin(in_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, out_channels)
        )
        self.mlp_successor = Seq(
            Lin(in_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, out_channels)
        )
        self.mlp_resources = Seq(
            Lin(out_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, out_channels)
        )
        self.mlp_same = Seq(
            Lin(in_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, self.hidden_channels), ELU(),
            Lin(self.hidden_channels, out_channels)
        )

    def forward(self, operations, resources, precedence_edges, requirement_edges):
        adj_ops = to_dense_adj(precedence_edges)[0]
        ops_idx_by_edges = requirement_edges[0]
        res_embeddings_by_edges = resources[requirement_edges[1]]
        agg_machine_embeddings = torch.zeros((operations.size(0), resources.size(1)))
        for i, op_idx in enumerate(ops_idx_by_edges):
            agg_machine_embeddings[op_idx] += res_embeddings_by_edges[i]
        predecessors = torch.zeros((operations.shape[0], self.embedding_size))
        successors = torch.zeros((operations.shape[0], self.embedding_size))
        for i in range(1, operations.shape[0] - 1):
            predecessors[i] = self.mlp_predecessor(operations[adj_ops[:,i].nonzero()].mean(dim=0))
            successors[i] = self.mlp_successor(operations[adj_ops[i].nonzero()].mean(dim=0))
        same_embeddings = self.mlp_same(operations[1:-1])
        agg_machine_embeddings = self.mlp_resources(agg_machine_embeddings[1:-1])
        embedding = torch.zeros((operations.shape[0], self.embedding_size))
        embedding[1:-1] = self.mlp_combined(torch.cat([predecessors[1:-1], successors[1:-1], agg_machine_embeddings, same_embeddings], dim=-1))
        return embedding
    
class HeterogeneousGAT(torch.nn.Module):
    def __init__(self):
        super(HeterogeneousGAT, self).__init__()
        embedding_size = GAT_CONF["embedding_dims"]
        self.resource_layers = torch.nn.ModuleList()
        self.operation_layers = torch.nn.ModuleList()
        self.resource_layers.append(ResourceAttentionEmbeddingLayer(len(RES_FEATURES), len(OP_FEATURES)))
        self.operation_layers.append(OperationEmbeddingLayer(len(OP_FEATURES), embedding_size))
        for _ in range(GAT_CONF["gnn_layers"]-1):
            self.resource_layers.append(ResourceAttentionEmbeddingLayer(embedding_size, embedding_size))
            self.operation_layers.append(OperationEmbeddingLayer(embedding_size, embedding_size))
        actor_critic_dim = GAT_CONF["actor_critic_dim"]
        self.actor_mlp = Seq(
            Lin(embedding_size * 4, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, 1)
        )
        self.critic_mlp = Seq(
            Lin(embedding_size * 2, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, actor_critic_dim), Tanh(), 
            Lin(actor_critic_dim, 1)
        )

    def forward(self, graph, actions):
        operations, resources = graph['operation'].x, graph['resource'].x
        precedence_edges = graph['operation', 'precedence', 'operation'].edge_index
        requirement_edges = graph['operation', 'uses', 'resource'].edge_index
        for l in range(GAT_CONF["gnn_layers"]):
            resources = self.resource_layers[l](resources, operations, requirement_edges)
            operations  = self.operation_layers[l](operations, resources, precedence_edges, requirement_edges)
        pooled_operations = global_mean_pool(operations, torch.zeros(operations.shape[0], dtype=torch.long))
        pooled_resources = global_mean_pool(resources, torch.zeros(resources.shape[0], dtype=torch.long))
        graph_state = torch.cat([pooled_operations, pooled_resources], dim=-1)[0]
        state_value = self.critic_mlp(graph_state)
        action_logits = []
        for op_idx, res_idx in actions:
            action_input = torch.cat([operations[op_idx], resources[res_idx], graph_state], dim=-1)
            action_logits.append(self.actor_mlp(action_input))
        action_logits = torch.stack(action_logits)
        action_probs = F.softmax(action_logits, dim=0)
        return action_probs, state_value

#====================================================================================================================
# =*= III. SOLVE AN INSTANCE =*=
#====================================================================================================================

def is_available(graph, res_idx, time):
    return graph['resource'].x[res_idx][RES_FEATURES["available_time"]].item() <= time

def scheduled(operation):
    return operation[OP_FEATURES["status"]] == SCHEDULED

def succ_one(edges, idx):
    target_edges = (edges[0] == idx).nonzero().view(-1)
    return edges[1, target_edges[0]].item() if target_edges.numel() > 0 else None
    
def pred_one(edges, idx):
    target_edges = (edges[1] == idx).nonzero().view(-1)
    return edges[0, target_edges[0]].item() if target_edges.numel() > 0 else None

def predecessor(graph, idx):
    edges = graph['operation', 'precedence', 'operation'].edge_index
    return pred_one(edges, idx)

def successors(graph, idx):
    succs = []
    edges = graph['operation', 'precedence', 'operation'].edge_index
    max = graph['operation'].x.shape[0] - 1
    end = False
    while not end:
        succ_idx = succ_one(edges, idx)
        if succ_idx==None or succ_idx>=max:
            end = True
        else:
            succs.append(succ_idx)
            idx = succ_idx
    return succs

def predecessors(graph, idx):
    preds = []
    edges = graph['operation', 'precedence', 'operation'].edge_index
    end = False
    while not end:
        pred_idx = pred_one(edges, idx)
        if pred_idx==None or pred_idx==0:
            end = True
        else:
            preds.append(pred_idx)
            idx = pred_idx
    return preds

def prune_requirements(graph, op_idx, res_idx):
    edge_type = ('operation', 'uses', 'resource')
    edges = graph[edge_type].edge_index
    mask = ~((edges[0] == op_idx) & (edges[1] == res_idx))
    graph[edge_type].edge_index = edges[:, mask]

def end_op(graph, idx):
    return op_val(graph, idx, 'start') + op_val(graph, idx, 'duration')

def to_schedule(graph, idx, t):
    op = graph['operation'].x[idx]
    pred_idx = predecessor(graph, idx)
    return (not scheduled(op)) and (pred_idx==None or pred_idx==0 or (scheduled(graph['operation'].x[pred_idx]) and t>=end_op(graph,pred_idx)))

def get_val(graph, type, idx, f_dic, feature):
    return graph[type].x[idx][f_dic[feature]].item()

def op_val(graph, idx, feature):
    return get_val(graph, 'operation', idx, OP_FEATURES, feature)

def res_val(graph, idx, feature):
    return get_val(graph, 'resource', idx, RES_FEATURES, feature)
    
def update_val(graph, type, idx, f_dic, feature, value):
    graph[type].x[idx][f_dic[feature]] = value

def update_op(graph, op_idx, updates):
    for feature, value in updates:
        update_val(graph, 'operation', op_idx, OP_FEATURES, feature, value)

def update_res(graph, res_idx, updates):
    for feature, value in updates:
        update_val(graph, 'resource', res_idx, RES_FEATURES, feature, value)

def possible_actions(graph, t):
    actions = []
    requirement_edges = graph['operation', 'uses', 'resource'].edge_index
    for op_idx, _ in enumerate(graph['operation'].x):
        if to_schedule(graph, op_idx, t):
            for res_idx in requirement_edges[1][requirement_edges[0] == op_idx]:
                if is_available(graph, res_idx, t):
                    actions.append((op_idx, res_idx.item()))
    return actions

def policy(probabilities, greedy=True):
    return torch.argmax(probabilities.view(-1)).item() if greedy else torch.multinomial(probabilities.view(-1), 1).item()

def solve(model, instance, train=False):
    graph, graph2instance, makespan = instance_to_graph(instance)
    sequences = [[] for _ in graph['resource'].x]
    utilization = [0 for _ in graph['resource'].x]
    nb_operations_to_schedule = graph['operation'].x.shape[0] - 2
    time = 0
    error = False
    requirements = graph['operation', 'uses', 'resource'].edge_index
    rewards = torch.Tensor([])
    values = torch.Tensor([])
    probabilities = []
    actions_idx = []
    states = []
    actions = []
    while nb_operations_to_schedule > 0 and not error:
        poss_actions = possible_actions(graph, time)
        if(len(poss_actions)>0):
            states.append(copy.deepcopy(graph))
            probs, state_value = model(copy.deepcopy(graph), poss_actions)
            values = torch.cat((values, torch.Tensor([state_value])))
            actions.append(poss_actions)
            probabilities.append(probs)
            idx = policy(probs, greedy=(not train))
            actions_idx.append(idx)
            op_idx, res_idx = poss_actions[idx]
            
            # 1. Update all operations of the selected job
            job_unscheduled_ops = op_val(graph, op_idx, "job_unscheduled_ops") - 1
            succs = successors(graph, op_idx)
            duration = op_val(graph, op_idx, "duration")
            end_job = time + duration
            update_res(graph, res_idx, [('available_time', end_job)])
            for succ in succs:
                update_op(graph, succ, [("start", end_job)])
                end_job = end_job + op_val(graph, succ, "duration")
            preds = predecessors(graph, op_idx)
            for op in preds + succs:
                update_op(graph, op, [("job_unscheduled_ops", job_unscheduled_ops), ("current_job_completion", end_job)])
            update_op(graph, op_idx, [("status", SCHEDULED), ("remaining_neighboring_resources", 1), ("start", time), ("job_unscheduled_ops", job_unscheduled_ops), ("current_job_completion", end_job)])
            sequences[res_idx].append(graph2instance[op_idx - 1])
            utilization[res_idx] = utilization[res_idx] + duration
            new_makespan = max(makespan, end_job)
            rewards = torch.cat((rewards, torch.Tensor([makespan - new_makespan])))
            makespan = new_makespan
            
            # 2. Update resources that have not been selected
            for other_res in requirements[1][requirements[0] == op_idx]:
                if other_res != res_idx:
                    remaining_neighboring_ops = res_val(graph, other_res, "remaining_neighboring_ops") - 1
                    update_res(graph, other_res, [('remaining_neighboring_ops', remaining_neighboring_ops)])
                    prune_requirements(graph, op_idx, other_res)
            nb_operations_to_schedule = nb_operations_to_schedule - 1
        else:
            # 3. Update resource usage and current time (based on next available machine and next free operation)
            next = -1
            for resoure in graph['resource'].x:
                available = resoure[RES_FEATURES["available_time"]].item()
                if available > time and (next < 0 or next > available):
                    next = available
            for operation in graph['operation'].x[1:-1]:
                available = operation[OP_FEATURES["start"]].item()
                if (not scheduled(operation)) and available > time and (next < 0 or next > available):
                    next = available
            if next > time:
                time = next
                for res_id, resoure in enumerate(graph['resource'].x):
                   resoure[RES_FEATURES["past_utilization_rate"]] = utilization[res_id] / time
            error = next < 0
    if train:
        return rewards, values, probabilities, states, actions, actions_idx
    else:
        return makespan, sequences   

#====================================================================================================================
# =*= IV. PROXIMAL POLICY OPTIMIZATION (PPO) =*=
#====================================================================================================================

def sample_batches(instances, batch_size=PPO_CONF['batch_size']):
    return random.sample(instances, batch_size)

def calculate_returns(rewards, gamma=PPO_CONF['discount_factor']):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def generalized_advantage_estimate(rewards, values, gamma=PPO_CONF['discount_factor'], lam=PPO_CONF['bias_variance_tradeoff']):
    GAE = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] - values[t]
        if t<len(rewards)-1:
            delta = delta + (gamma * values[t+1])
        GAE = delta + gamma * lam * GAE
        advantages.insert(0, GAE)
    return advantages

def PPO_loss(model, old_probs, states, actions, actions_idx, advantages, old_values, returns, clip_ratio=PPO_CONF['clip_ratio'], actor_w=PPO_CONF['policy_loss'], critic_w=PPO_CONF['value_loss'], entropy_w=PPO_CONF['entropy']):
    new_log_probs = torch.Tensor([])
    old_log_probs = torch.Tensor([])
    for i in range(len(states)):
        p,_ = model(states[i], actions[i])
        a = actions_idx[i]
        new_log_probs = torch.cat((new_log_probs, torch.log(p[a])))
        old_log_probs = torch.cat((old_log_probs, torch.log(old_probs[i][a])))
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = torch.mean(torch.stack([(V_old - r) ** 2 for V_old, r in zip(old_values, returns)]))
    entropy_loss = -new_log_probs.mean()
    print("value loss - "+str(value_loss))
    print("policy loss - "+str(policy_loss)) 
    print("entropy loss - "+str(entropy_loss)) 
    print("======")
    return (actor_w*policy_loss) + (critic_w*value_loss) - (entropy_loss*entropy_w)

def PPO_optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def PPO_train(instances, batch_size=PPO_CONF['batch_size'], iterations=PPO_CONF['train_iterations'], validation_rate=PPO_CONF['validation_rate'], switch_batch=PPO_CONF['switch_batch'], validation_ratio=PPO_CONF['validation_ratio'], epochs=PPO_CONF['opt_epochs']):
    model = HeterogeneousGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPT_CONF['learning_rate'])
    model.train()
    random.shuffle(instances)
    num_val = int(len(instances) * validation_ratio)
    train_instances, val_instances = instances[num_val:], instances[:num_val]
    for iteration in range(iterations):
        if iteration % switch_batch == 0:
            current_batch = sample_batches(train_instances, batch_size)
        all_rewards, all_values, all_probabilities, all_states, all_actions, all_actions_idx = [], [], [], [], [], []
        for instance in current_batch:
            rewards, values, probabilities, states, actions, actions_idx = solve(model, instance, train=True)
            all_rewards.append(rewards)
            all_values.append(values)
            all_probabilities.extend(probabilities)
            all_states.extend(states)
            all_actions.extend(actions)
            all_actions_idx.extend(actions_idx)
        all_returns = [ri for r in all_rewards for ri in calculate_returns(r)]
        advantages = torch.Tensor([gae for r, v in zip(all_rewards, all_values) for gae in generalized_advantage_estimate(r, v)])
        flattened_values = [v for vals in all_values for v in vals]
        for _ in range(epochs):
            loss = PPO_loss(model, all_probabilities, all_states, all_actions, all_actions_idx, advantages, flattened_values, all_returns)
            PPO_optimize(optimizer, loss)
        if iteration % validation_rate == 0:
            validate(model, val_instances)
    return model

def validate(model, instances):
    model.eval()
    total_rewards = 0
    total_loss = 0
    with torch.no_grad():
        for instance in instances:
            rewards, values, probabilities, states, actions, actions_idx = solve(model, instance, train=True)
            loss = PPO_loss(model, probabilities, states, actions, actions_idx , generalized_advantage_estimate(rewards, values), values, calculate_returns(rewards))
            total_rewards += sum(rewards).item()
            total_loss += loss.item()
    num_instances = len(instances)
    avg_reward = total_rewards / num_instances
    avg_loss = total_loss / num_instances
    print(f'Validation - Average Reward: {avg_reward:.4f}, Average Loss: {avg_loss:.4f}')
    model.train()

def test(model, instances, optimals):
    errors = np.array([])
    nbr_optimals = 0
    for idx, instance in enumerate(instances):
        makespan, sequences = solve(model, instance, train=False)
        optimal =  optimals.iloc[idx]['values']
        error = (makespan - optimal)/optimal
        if error <= 0:
            nbr_optimals = nbr_optimals + 1
        errors = np.append(errors, error)
    return nbr_optimals, errors

#====================================================================================================================
# =*= V. EXECUTE THE COMPLETE CODE =*=
#====================================================================================================================

train_instances = load_instances(TRAIN_INSTANCES_PATH)
test_instances = load_instances(TEST_INSTANCES_PATH)
test_optimal = pd.read_csv(TEST_INSTANCES_PATH+'optimal.csv')
print("Starting training process...")
model = PPO_train(train_instances)
print("Starting tests...")
nbr_optimals, errors = test(model, test_instances, test_optimal)
print("Optimal makespans: ", test_optimal['values'].values)
print("Errors (as percentages): ", errors)
print("Maximum error: ", np.max(errors))
print("Minimum error: ",  np.min(errors))
print("Mean error: ",  np.mean(errors))
print("Number of optimal values: ",  nbr_optimals)