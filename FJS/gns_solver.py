import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from common import load_instances, OP_STRUCT
from torch.nn import Sequential as Seq, Linear as Lin, ELU
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
OP_FEATURES = {"status": 0, "remaining_neighboring_resources": 1, "duration": 2, "start": 3, "job_unscheduled_ops": 4, "current_job_completion": 5}
RES_FEATURES = {"available_time": 0, "remaining_neighboring_ops": 1, "past_utilization_rate": 2}
GAT_CONF = {"gnn_layers": 2, "embedding_dims": 8, "MLP_size": 128, "heads": {"layers": 2, "init_dims": 64}}
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

class OperationEmbeddingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(OperationEmbeddingLayer, self).__init__(aggr='none')
        hidden_channels = GAT_CONF["MLP_size"]
        self.mlp_combined = Seq(
            Lin(out_channels, hidden_channels), ELU(),
            Lin(hidden_channels, hidden_channels), ELU(),
            Lin(hidden_channels, out_channels)
        )
        self.mlp_predecessor = Seq(
            Lin(in_channels, hidden_channels), ELU(),
            Lin(hidden_channels, hidden_channels), ELU(),
            Lin(hidden_channels, out_channels)
        )
        self.mlp_successor = Seq(
            Lin(in_channels, hidden_channels), ELU(),
            Lin(hidden_channels, hidden_channels), ELU(),
            Lin(hidden_channels, out_channels)
        )
        self.mlp_resources = Seq(
            Lin(out_channels, hidden_channels), ELU(),
            Lin(hidden_channels, hidden_channels), ELU(),
            Lin(hidden_channels, out_channels)
        )
        self.mlp_same = Seq(
            Lin(in_channels, hidden_channels), ELU(),
            Lin(hidden_channels, hidden_channels), ELU(),
            Lin(hidden_channels, out_channels)
        )

    def forward(self, operations, resources, precedence_edges, requirement_edges):
        adj_ops = to_dense_adj(precedence_edges)[0]
        agg_machine_embeddings = to_dense_adj(requirement_edges)[0].matmul(v)
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
    
class ResourceEmbeddingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ResourceEmbeddingLayer, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out
    
class HeterogeneousGAT(torch.nn.Module):
    def __init__(self):
        super(HeterogeneousGAT, self).__init__()
        embedding_size = GAT_CONF["embedding_dims"]
        self.resource_layers = torch.nn.ModuleList()
        self.resource_layers.append(ResourceEmbeddingLayer(len(RES_FEATURES), embedding_size))
        for _ in range(1, range(GAT_CONF["gnn_layers"])):
            self.resource_layers.append(ResourceEmbeddingLayer(embedding_size, embedding_size))
        self.operation_layers = torch.nn.ModuleList()
        self.resource_layers.append(OperationEmbeddingLayer(len(OP_FEATURES), embedding_size))
        for _ in range(1, range(GAT_CONF["gnn_layers"])):
            self.resource_layers.append(OperationEmbeddingLayer(embedding_size, embedding_size))

    def forward(self, graph):
        operations, resources = graph['operation'].x, graph['resource'].x
        precedence_edges = graph['operation', 'precedence', 'operation'].edge_index
        requirement_for_op_edges = graph['operation', 'uses', 'resource'].edge_index
        requirement_for_res_edges = graph['resource', 'execute', 'operation'].edge_index
        for l in range(GAT_CONF["gnn_layers"]):
            graph['resource'].x = self.resource_layers[l](resources, requirement_for_res_edges)
            graph['operation'].x  = self.operation_layers[l](operations, resources, precedence_edges, requirement_for_op_edges)
        return graph

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