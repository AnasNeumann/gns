import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool
from common import load_instances, OP_STRUCT
from torch.nn import Sequential as Seq, Linear as Lin, ELU, Tanh, Parameter
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import copy

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
OP_FEATURES = {"status": 0, "remaining_neighboring_resources": 1, "duration": 2, "start": 3, "job_unscheduled_ops": 4, "current_job_completion": 5}
RES_FEATURES = {"available_time": 0, "remaining_neighboring_ops": 1, "past_utilization_rate": 2}
GAT_CONF = {"gnn_layers": 2, "embedding_dims": 8, "MLP_size": 128, "actor_critic_dim": 64}
PPO_CONF = {"train_iterations": 1000, "opt_iterations": 3, "batch_size": 20, "clip_ratio": 0.2, "policy_loss": 1, "value_loss": 0.5, "entropy": 0.01, "discount_factor": 1.0}
OPT_CONF = {"learning_rate": 2e-4}

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

    resource_id = 0
    for type, quantity in enumerate(resources):
        res_of_type = []
        for _ in range(quantity):
            graph = add_node(graph, 'resource', torch.tensor([[0, init_ops_by_resource(type, jobs), 0]], dtype=torch.float))
            res_of_type.append(resource_id)
            resource_id += 1
        resources_by_type.append(res_of_type)
    
    op_id = 1
    for job in jobs:
        first_op_id = op_id
        ops2graph = []
        for position, operation in enumerate(job):
            duration = operation[OP_STRUCT["duration"]]
            available_resources = resources[operation[OP_STRUCT["resource_type"]]]
            graph = add_node(graph, 'operation', torch.tensor([[NOT_SCHEDULED, available_resources, duration, init_start_time(job, position), len(job), init_end_time(job, len(job)-1)]], dtype=torch.float))
            ops2graph.append(op_id)
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
    return graph

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

    def forward(self, graph, actions, t):
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
# =*= III. FUNCTION TO USE THE SCHEDULING GNN AND UPDATE THE GRAPH =*=
#====================================================================================================================

def is_available(graph, res_idx, time):
    return graph['resource'].x[res_idx][RES_FEATURES["available_time"]].item() <= time

def scheduled(operation):
    return operation[OP_FEATURES["status"]] != NOT_SCHEDULED

def predecessor(graph, idx):
    edges = graph['operation', 'precedence', 'operation'].edge_index
    target_edges = (edges[1] == idx).nonzero().view(-1)
    if target_edges.numel() > 0:
        return edges[0, target_edges[0]].item() 
    else:
        return None

def to_schedule(graph, idx):
    op = graph['operation'].x[idx]
    pred_idx = predecessor(graph, idx)
    return (not scheduled(op)) and (pred_idx==None or pred_idx==0 or scheduled(graph['operation'].x[pred_idx]).item())

def possible_actions(graph, t):
    actions = []
    requirement_edges = graph['operation', 'uses', 'resource'].edge_index
    for op_idx, _ in enumerate(graph['operation'].x):
        if to_schedule(graph, op_idx):
            for res_idx in requirement_edges[1][requirement_edges[0] == op_idx]:
                if is_available(graph, res_idx, t):
                    actions.append((op_idx, res_idx.item()))
    return actions

#====================================================================================================================
# =*= IV. PROXIMAL POLICY OPTIMIZATION (PPO) DEEP-REINFORCEMENT ALGORITHM =*=
#====================================================================================================================

# TODO Une fonction qui ordonnance les jobs les uns après les autres
# Créer une liste des combinaisons possibles comme dans le GNN pour comparer
# Update is_schedule, start_time, remaining_neighboring_resources for one operation only but job_unscheduled_ops and current_job_completion for all job operations
# Update available_time and past_utilization_rate for the selected resource
# Update remaining_neighboring_ops for the not selected resources
# [Maybe] update past_utilization_rate for all resources
# Update the final Makespan
# Update current time "t" based on available resources and operations!
# Add feature normalization 

def solve(instance, train=False):
    graph = instance_to_graph(instance)
    model = HeterogeneousGAT()
    CURRENT_TIME = 0
    SOLVED = False
    while not SOLVED:
        actions = possible_actions(graph, CURRENT_TIME)
        print(actions)
        action_probs = model(copy.deepcopy(graph), actions, CURRENT_TIME)
        print(action_probs)
        SOLVED = True

#====================================================================================================================
# =*= V. EXECUTE THE COMPLETE CODE =*=
#====================================================================================================================

train_instances = load_instances(TRAIN_INSTANCES_PATH)
test_instances = load_instances(TEST_INSTANCES_PATH)
print(train_instances[0])
solve(train_instances[0])

#torch.save(model.state_dict(), 'GNS.pth')
#model_loaded = HeterogeneousGAT(1)
#model_loaded.load_state_dict(torch.load('GNS.pth'))