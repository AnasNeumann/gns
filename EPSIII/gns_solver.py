import argparse
import pickle
from torch.multiprocessing import Pool, set_start_method
import os
from model import Instance, GraphInstance, L1_ACTOR_CRITIC_GNN, L1_EMBEDDING_GNN
from common import load_instance, to_bool, features2tensor, id2tensor
import torch
torch.autograd.set_detect_anomaly(True)

PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
OUTSOURCING = "outsourcing"
SCHEDULING = "scheduling"
MATERIAL_USE = "material_use"
LEARNING_RATE = 2e-4
PARRALLEL = True
DEVICE = None
PPO_CONF = {
    "validation_rate": 10,
    "switch_batch": 20,
    "train_iterations": 1000, 
    "opt_epochs": 3,
    "batch_size": 20,
    "clip_ratio": 0.2,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.01,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation_ratio': 0.1
}
GNN_CONF = {
    'embedding_size': 16,
    'nb_layers': 2,
    'hidden_channels': 128
}
AC_CONF = {
    'hidden_channels': 64
}

# =====================================================
# =*= MODEL LOADING METHOD =*=
# =====================================================

def save_models(models):
    for model, name in models:
        torch.save(model.state_dict(), './models/'+name+'_weights.pth')

def load_trained_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load('./models/gnn_weights.pth'))
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    outsourcing_actor.load_state_dict(torch.load('./models/outsourcing_weights.pth'))
    scheduling_actor.load_state_dict(torch.load('./models/scheduling_weights.pth'))
    material_actor.load_state_dict(torch.load('./models/material_weights.pth'))
    return (shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')

def init_new_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    return (shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')

# =====================================================
# =*= TRANSLATE INSTANCE TO GRAPH =*=
# =====================================================

def build_item(i: Instance, graph: GraphInstance, p, e, head):
    design_time, physical_time = i.item_processing_time(p, e)
    children_design_time = 0
    max_children_time = 0
    childrens = i.get_children(p, e, False)
    childrens_physical_operations =0
    for children in childrens:
        cdt, cpt = i.item_processing_time(p, children)
        children_design_time += cdt
        max_children_time = cdt+cpt if (cdt+cpt > max_children_time) else max_children_time
        start_c, end_c = i.get_operations_idx(p, children)
        for child_op in range(start_c, end_c):
            if not i.is_design[p][child_op]:
                childrens_physical_operations += 1
    parents_design_time = 0
    parents_physical_time = 0
    ancestors = i.get_ancestors(p, e)
    for ancestor in ancestors:
            adt, apt = i.item_processing_time(p, ancestor)
            parents_physical_time += apt
            parents_design_time += adt
    total_end = design_time + parents_design_time + physical_time + max_children_time
    item_id = graph.add_item(e, head, i.external[p][e], -1, i.external_cost[p][e], i.outsourcing_time[p][e], physical_time, design_time, len(ancestors), len(childrens), parents_physical_time, children_design_time, parents_design_time, total_end)
    op_start = parents_design_time
    start, end = i.get_operations_idx(p,e)
    for o in range(start, end):
        succs = end-(o+1)
        minutes = not (i.in_hours[p][o] or i.in_days[p][o])
        operation_time = i.operation_time(p,o)
        op_id = graph.add_operation(o, i.is_design[p][o], i.simultaneous[p][o], minutes, i.in_hours[p][o], i.in_days[p][o], succs, childrens_physical_operations + succs, operation_time, i.required_rt(p,o), -1, op_start, op_start+operation_time)
        graph.add_operation_assembly(item_id, op_id)
        for r in i.required_resources(p,o):
            if i.finite_capacity[r]:
                graph.add_need_for_resources(op_id, graph.resource_i2g(r), [0, i.execution_time[r][p][o], i.execution_time[r][p][o], op_start, op_start+i.execution_time[r][p][o]])
            else:
                graph.add_need_for_materials(op_id, graph.material_i2g(r), [0, op_start, i.quantity_needed[r][p][o]])
    for children in i.get_children(p, e, True):
        graph, child_id = build_item(i, graph, p, children, False)
        graph.add_item_assembly(item_id, child_id)
    return graph, item_id

def translate(i: Instance):
    graph = GraphInstance()
    for rt in range(i.nb_HR_types):
        resources = i.resources_by_type(rt)
        operations = i.operations_by_resource_type(rt)
        res_idx = []
        for r in resources:
            if i.finite_capacity[r]:
                res_id = graph.add_resource(r, 0, 0, 0, len(operations), len(resources))
                if len(res_idx)>0:
                    for other_id in res_idx:
                        graph.add_same_types(other_id, res_id)
                res_idx.append(res_id)
            else:
                remaining_quantity_needed = 0
                for p, o in operations:
                    remaining_quantity_needed += i.quantity_needed[r][p][o]
                graph.add_material(r, i.init_quantity[r], i.purchase_time[r], remaining_quantity_needed)
            res_idx += 1
    for p in range(i.get_nb_projects()):
        head = i.project_head(p)
        graph = build_item(i, graph, p, head, True)
    return graph

# =====================================================
# =*= EXECUTE ONE INSTANCE =*=
# =====================================================

def solve_one(instance: Instance, graph: GraphInstance, shared_GNN: L1_EMBEDDING_GNN, outsourcing_actor: L1_ACTOR_CRITIC_GNN, scheduling_actor: L1_ACTOR_CRITIC_GNN, material_actor: L1_ACTOR_CRITIC_GNN, path, save=False):
    # TODO solve and save results
    pass

# =====================================================
# =*= PPO TRAINING PROCESS =*=
# =====================================================

def load_training_dataset():
    instances = []
    for size in PROBLEM_SIZES:
        problems = []
        path = './instances/train/'+size+'/'
        for i in os.listdir(path):
            if i.endswith('.pkl'):
                file_path = os.path.join(path, i)
                with open(file_path, 'rb') as file:
                    problems.append(pickle.load(file))
        instances.append(problems)
    print("end of loading!")
    return instances

def train(instances, shared_GNN: L1_EMBEDDING_GNN, outsourcing_actor: L1_ACTOR_CRITIC_GNN, scheduling_actor: L1_ACTOR_CRITIC_GNN, material_actor: L1_ACTOR_CRITIC_GNN):
    # TODO train models based on instances then save models
    save_models([shared_GNN, outsourcing_actor, scheduling_actor, material_actor])

# =====================================================
# =*= MAIN CODE =*=
# =====================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    if args.mode == 'test':
        PPO_CONF['train_iterations'] = 10
        PPO_CONF['batch_size'] = 3
        PARRALLEL = False
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    if to_bool(args.train):
        print("LOAD DATASET TO TRAIN MODELS")
        instances = load_training_dataset()
        print("TRAIN MODELS WITH PPO")
        shared_GNN, outsourcing_actor, scheduling_actor, material_actor = init_new_models()
        train(instances, shared_GNN, outsourcing_actor, scheduling_actor, material_actor)
    else:
        print("SOLVE A TARGET INSTANCE")
        INSTANCE_PATH = './instances/test/'+args.size+'/instance_'+args.id+'.pkl'
        SOLUTION_PATH = './instances/test/'+args.size+'/solution_'+args.id+'.csv'
        print("Loading "+INSTANCE_PATH+"...")
        instance = load_instance(INSTANCE_PATH)
        shared_GNN, outsourcing_actor, scheduling_actor, material_actor = load_trained_models()
        graph = translate(instance)
        solve_one(instance, graph, shared_GNN, outsourcing_actor, scheduling_actor, material_actor, SOLUTION_PATH, save=True)
    print("===* END OF FILE *===")