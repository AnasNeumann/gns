import torch
from torch_geometric.data import Data, HeteroData
from common import load_instances, OP_STRUCT

# Configuration
TRAIN_INSTANCES_PATH = './FJS/instances/train/'
TEST_INSTANCES_PATH = './FJS/instances/test/'
NOT_SCHEDULED = 0
SCHEDULED = 1
INITIAL_MAKESPAN = 0
OPERATION_FEATURES = {"status": 0, "num_resources": 1, "duration": 2, "start": 3, "num_ops": 4, "job_completion": 5} # TODO 1. NEED TO DO THE POSITION AS A PERCENTAGE
RESOURCE_FEATURES = {"available": 0, "num_ops": 1, "utilization": 2}

# TODO 2. MAYBE WE WILL HAVE TO CHANGE NEIGHBOORING OPERATIONS AND RESSOURCES COMPUTATION/METHOD WHEN SCHEDULED!!
# TODO 3. NEED TO DO THE RESOURCE UTILIZATION METHOD
# TODO 4?. NEED FOR GLOBAL INFORMATION: NBR RESOURCES, NBR JOBS, NBR OPERATIONS, etc.?

#====================================================================================================================
# =*= FONCTIONS TO BUILD AND UPDATE THE RAW GRAPH =*=
#====================================================================================================================

# Get the job completion time in the current schedule
def job_current_completion(graph, operations_idx):
    completion = 0
    for id in operations_idx:
        if is_scheduled(graph, id):
            _,_,end_time = get_operation_dates(graph, id)
            completion = max(completion, end_time)
    return completion

# Get the utilization rate of a specific resource
def res_utilization(resource, current_time):
    return 0

# Get the number of operations that need a type of resources
def ops_by_resource(type, jobs):
    return len([op for job in jobs for op in job if op[OP_STRUCT["resource_type"]]==type])

# Get start and end dates of an operation
def get_operation_dates(graph, op_id):
    start = OPERATION_FEATURES["start"]
    duration = OPERATION_FEATURES["duration"]
    return graph['operation'].x[op_id,start].item(), graph['operation'].x[op_id,duration].item(), graph['operation'].x[op_id,start].item()+graph['operation'].x[op_id,duration].item()

# Check if an operation has been scheduled already
def is_scheduled(graph, op_id):
    return graph['operation'].x[op_id, OPERATION_FEATURES["status"]] == SCHEDULED

# A function to estimate the earliest possible start time
def estimate_start_time(graph, prec_operations_idx):
    if len(prec_operations_idx)==0:
        return 0
    else:
        last_scheduled_pos = -1
        max_end_time = 0
        for pos, prec_id, in enumerate(prec_operations_idx):
            if is_scheduled(graph, prec_id):
                _,_,pred_end_time = get_operation_dates(graph, prec_id)
                max_end_time = pred_end_time
                last_scheduled_pos = pos
            else:
                break
        if(last_scheduled_pos < len(prec_operations_idx)-1):
            for prec_id in prec_operations_idx[last_scheduled_pos+1:]:
                _,duration,_ = get_operation_dates(graph, prec_id)
                max_end_time = max_end_time + duration
        return max_end_time

# Add a new node in the graph
def add_node(graph, type, features):
    graph[type].x = torch.cat([graph[type].x, features], dim=0) if type in graph.node_types else features
    return graph

# Add a new edge in the graph
def add_edge(graph, first, middle, last, features):
    graph[first, middle, last].edge_index = torch.cat([graph[first, middle, last].edge_index, features], dim=1) if (first, middle, last) in graph.edge_types else features
    return graph

# Function to build the graph instance
def instance_to_graph(instance):
    graph = HeteroData()
    resources = instance['resources']
    jobs = instance['jobs']
    zero_features = torch.zeros((1, 6), dtype=torch.int64)
    graph = add_node(graph, 'operation', zero_features) # start node
    operations2graph = [] 
    resources_by_type = []

    # I. Resource nodes
    resource_id = 0
    for type, quantity in enumerate(resources):
        res_of_type = []
        for resource in range(quantity):
            graph = add_node(graph, 'resource', torch.tensor([[INITIAL_MAKESPAN, ops_by_resource(type, jobs), res_utilization(resource, INITIAL_MAKESPAN)]]))
            res_of_type.append(resource_id)
            resource_id += 1
        resources_by_type.append(res_of_type)
    
    # II. Operation nodes, precedence, and start/end edges
    op_id = 1
    for job in jobs:
        first_op_id = op_id
        prec_operations_idx = []
        ops2graph = []
        for operation in job:
            graph = add_node(graph, 'operation', torch.tensor([[NOT_SCHEDULED, resources[operation[OP_STRUCT["resource_type"]]], operation[OP_STRUCT["duration"]], estimate_start_time(graph, prec_operations_idx), len(job), INITIAL_MAKESPAN]]))
            prec_operations_idx.append(op_id)
            ops2graph.append(op_id)
            if op_id > first_op_id: # III. Precedence edges (1st relation)
                graph = add_edge(graph, 'operation', 'precedence', 'operation', torch.tensor([[op_id - 1], [op_id]], dtype=torch.long))
            op_id += 1
        operations2graph.append(ops2graph)
    
    # IV. Precedence edges with dummy start and end nodes
    graph = add_node(graph, 'operation', zero_features) # end node
    op_id = 1
    for job in jobs:
        graph = add_edge(graph, 'start', 'to', 'operation', torch.tensor([[0], [op_id]], dtype=torch.long))
        op_id += len(job)
        graph = add_edge(graph, 'operation', 'to', 'end', torch.tensor([[op_id-1], [instance['size']+1]], dtype=torch.long))

    # V. Resource compatibility and use edges (2nd relation)
    for job_id, job in enumerate(jobs):
        for op_id, operation in enumerate(job):
            op2graph_id = operations2graph[job_id][op_id]
            for res2graph_id in resources_by_type[operation[OP_STRUCT["resource_type"]]]:
                graph = add_edge(graph, 'operation', 'uses', 'resource', torch.tensor([[op2graph_id], [res2graph_id]], dtype=torch.long))            
    return graph, operations2graph, resources_by_type

# Display the details of the current graph
def display_graph(graph):
    print("Graph overview: " + str(graph))
    print("Resources: " + str(graph['resource']))
    print("Operations: " + str(graph['operation']))
    print("Precedence relations: " + str(graph['operation', 'precedence', 'operation']))
    print("Link with dummy start node: " + str(graph['start', 'to', 'operation']))
    print("Link with dummy end node: " + str(graph['operation', 'to', 'end']))
    print("Requirements: " + str(graph['operation', 'uses', 'resource']))

#====================================================================================================================
# =*= GRAPH ATTENTION NEURAL NET ARCHITECTURE =*=
#====================================================================================================================

#====================================================================================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) DEEP-REINFORCEMENT ALGORITHM =*=
#====================================================================================================================

#====================================================================================================================
# =*= EXECUTE THE COMPLETE CODE =*=
#====================================================================================================================

train_instances = load_instances(TRAIN_INSTANCES_PATH)
test_instances = load_instances(TEST_INSTANCES_PATH)
print(train_instances[0])
graph,_,_ = instance_to_graph(train_instances[0])
display_graph(graph)