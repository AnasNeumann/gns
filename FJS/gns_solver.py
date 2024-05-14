import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from common import load_instances, OP_STRUCT
from torch_geometric.utils import to_undirected

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
OPERATION_FEATURES = {"status": 0, "remaining_neighboring_resources": 1, "duration": 2, "start": 3, "job_unscheduled_ops": 4, "current_job_completion": 5}
RESOURCE_FEATURES = {"available_time": 0, "remaining_neighboring_ops": 1, "past_utilization_rate": 2}

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
                graph = add_edge(graph, 'resource', 'execute', 'operation', torch.tensor([[res2graph_id], [op2graph_id]], dtype=torch.long))        
    return graph

def display_graph(graph):
    print("Graph_overview: " + str(graph))
    print("Resources: " + str(graph['resource']))
    print("Operations: " + str(graph['operation']))
    print("Precedence_relations: " + str(graph['operation', 'precedence', 'operation']))
    print("Requirements operation->resource: " + str(graph['operation', 'uses', 'resource']))
    print("Requirements resource->operation: " + str(graph['resource', 'execute', 'operation']))

#====================================================================================================================
# =*= II. GRAPH ATTENTION NEURAL NET ARCHITECTURE =*=
#====================================================================================================================

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