import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool
from common import load_instances, OP_STRUCT
from torch.nn import Sequential as Seq, Linear as Lin, ELU, Tanh, Parameter
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
OP_FEATURES = {"status": 0, "remaining_neighboring_resources": 1, "duration": 2, "start": 3, "job_unscheduled_ops": 4, "current_job_completion": 5}
RES_FEATURES = {"available_time": 0, "remaining_neighboring_ops": 1, "past_utilization_rate": 2}
GAT_CONF = {"gnn_layers": 2, "embedding_dims": 8, "MLP_size": 128, "attention_heads": 3, "actor_critic_dim": 64}
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
            graph = add_node(graph, 'resource', torch.tensor([[0, init_ops_by_resource(type, jobs), 0]]))
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
            graph = add_node(graph, 'operation', torch.tensor([[NOT_SCHEDULED, available_resources, duration, init_start_time(job, position), len(job), init_end_time(job, len(job)-1)]]))
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
    def __init__(self, resource_features_dim, operation_features_dim, num_heads=GAT_CONF["attention_heads"]):
        super(ResourceAttentionEmbeddingLayer, self).__init__(aggr='add', node_dim=0)
        self.num_heads = num_heads
        self.hidden_dim = GAT_CONF["embedding_dims"]
        self.resource_transform = Lin(resource_features_dim, self.hidden_dim)
        self.operation_transform = Lin(operation_features_dim, self.hidden_dim)
        self.att_resource_self = Parameter(torch.Tensor(1, num_heads, 2 * self.hidden_dim))
        self.att_resource_operation = Parameter(torch.Tensor(1, num_heads, 2 * self.hidden_dim))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)  # LeakyReLU activation
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att_resource_self)
        torch.nn.init.xavier_uniform_(self.att_resource_operation)

    def forward(self, resources, operations, requirement_edges):
        resources = self.repeat(1, self.num_heads).resource_transform(resources).view(-1, self.num_heads, self.hidden_dim)
        operations = self.repeat(1, self.num_heads).operation_transform(operations).view(-1, self.num_heads, self.hidden_dim)
        self_attention = self.leaky_relu((resources * resources).sum(dim=-1))
        cross_attention = self.propagate(requirement_edges, x=operations, v=resources)
        alpha = self.softmax(torch.cat([self_attention.unsqueeze(-1), cross_attention], dim=-1), index=requirement_edges[0])
        normalized_self_coef, normalized_operations_coef = alpha.split([1, alpha.size(-1) - 1], dim=-1)
        v_prime = torch.nn.functional.elu((normalized_self_coef * resources).sum(dim=-2) + (normalized_operations_coef * cross_attention).sum(dim=-2))
        return v_prime

    def message(self, x_j, v_i):
        att = self.leaky_relu((self.att_resource_operation * torch.cat([v_i, x_j], dim=-1)).sum(dim=-1))
        return att.unsqueeze(-1) * x_j

    def update(self, aggr_out):
        return aggr_out
    
class OperationEmbeddingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(OperationEmbeddingLayer, self).__init__(aggr='none')
        self.hidden_channels = GAT_CONF["MLP_size"]
        self.mlp_combined = Seq(
            Lin(out_channels, self.hidden_channels), ELU(),
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
        agg_machine_embeddings = to_dense_adj(requirement_edges)[0].matmul(resources)
        predecessors = torch.zeros_like(operations)
        successors = torch.zeros_like(operations)
        for i in range(1, operations.shape[0] - 1):
            preds = adj_ops[:,i].nonzero()
            succs = adj_ops[i].nonzero()
            if preds.shape[0] > 0: 
                predecessors[i] = self.mlp_predecessor(operations[preds].mean(dim=0))
            if succs.shape[0] > 0:
                successors[i] = self.mlp_successor(operations[succs].mean(dim=0))
        x_prime = operations.clone()
        x_prime[1:-1] = self.mlp_combined(F.elu(torch.cat([predecessors, successors, self.mlp_resources(agg_machine_embeddings[1:-1]), self.mlp_same(operations[1:-1])], dim=-1)))
        return x_prime

    def message(self, x_j):
        return x_j
    
class HeterogeneousGAT(torch.nn.Module):
    def __init__(self, possible_decisions=1):
        super(HeterogeneousGAT, self).__init__()
        embedding_size = GAT_CONF["embedding_dims"]
        self.resource_layers = torch.nn.ModuleList()
        self.resource_layers.append(ResourceAttentionEmbeddingLayer(len(RES_FEATURES), embedding_size))
        for _ in range(1, range(GAT_CONF["gnn_layers"])):
            self.resource_layers.append(ResourceAttentionEmbeddingLayer(embedding_size, embedding_size))
        self.operation_layers = torch.nn.ModuleList()
        self.resource_layers.append(OperationEmbeddingLayer(len(OP_FEATURES), embedding_size))
        for _ in range(1, range(GAT_CONF["gnn_layers"])):
            self.resource_layers.append(OperationEmbeddingLayer(embedding_size, embedding_size))
        actor_critic_dim = GAT_CONF["actor_critic_dim"]
        self.actor_mlp = Seq(
            Lin(embedding_size * 4, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, possible_decisions)
        )
        self.critic_mlp = Seq(
            Lin(embedding_size * 2, actor_critic_dim), Tanh(),
            Lin(actor_critic_dim, actor_critic_dim), Tanh(), 
            Lin(actor_critic_dim, 1)
        )

    def forward(self, graph):
        operations, resources = graph['operation'].x, graph['resource'].x
        precedence_edges = graph['operation', 'precedence', 'operation'].edge_index
        requirement_edges = graph['operation', 'uses', 'resource'].edge_index
        for l in range(GAT_CONF["gnn_layers"]):
            resources = self.resource_layers[l](resources, requirement_edges)
            operations  = self.operation_layers[l](operations, resources, precedence_edges, requirement_edges)
        pooled_operations = global_mean_pool(operations, torch.zeros(operations.shape[0], dtype=torch.long))
        pooled_resources = global_mean_pool(resources, torch.zeros(resources.shape[0], dtype=torch.long))
        graph_state = torch.cat([pooled_operations, pooled_resources], dim=-1)
        state_value = self.critic_mlp(graph_state)
        action_logits = []
        for i, op_embedding in enumerate(operations):
            # TODO add two additional constraints: op availability and res availability!
            feasible_resource_indices = requirement_edges[1][requirement_edges[0] == i]
            feasible_resources = resources[feasible_resource_indices]
            for resource_embedding in feasible_resources:
                action_input = torch.cat([op_embedding, resource_embedding, graph_state], dim=-1)
                action_logits.append(self.actor_mlp(action_input))
        action_logits = torch.stack(action_logits)
        action_probs = F.softmax(action_logits, dim=0)
        return action_probs, state_value

#====================================================================================================================
# =*= III. FUNCTION TO USE THE SCHEDULING GNN AND UPDATE THE GRAPH =*=
#====================================================================================================================

#====================================================================================================================
# =*= IV. PROXIMAL POLICY OPTIMIZATION (PPO) DEEP-REINFORCEMENT ALGORITHM =*=
#====================================================================================================================

#====================================================================================================================
# =*= V. EXECUTE THE COMPLETE CODE =*=
#====================================================================================================================

train_instances = load_instances(TRAIN_INSTANCES_PATH)
test_instances = load_instances(TEST_INSTANCES_PATH)
print(train_instances[0])
graph = instance_to_graph(train_instances[0])
display_graph(graph)
model = HeterogeneousGAT(1)
torch.save(model.state_dict(), 'GNS.pth')
model_loaded = HeterogeneousGAT(1)
model_loaded.load_state_dict(torch.load('GNS.pth'))
print(model_loaded)