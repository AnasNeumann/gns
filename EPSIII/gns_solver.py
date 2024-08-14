import argparse
import pickle
from torch.multiprocessing import Pool, set_start_method
import os
from model import Instance, GraphInstance, L1_ACTOR_CRITIC_GNN, L1_EMBEDDING_GNN
from common import load_instance, to_bool, features2tensor, id2tensor, to_binary
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
NOT_YET = -1
YES = 1
NO = 0

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

def build_item(i: Instance, graph: GraphInstance, p, e, head=False):
    design_time, physical_time = i.item_processing_time(p, e)
    children_design_time = 0
    max_children_time = 0
    childrens = i.get_children(p, e, False)
    childrens_physical_operations =0
    for children in childrens:
        cdt, cpt = i.item_processing_time(p, children)
        children_design_time += cdt
        max_children_time = max(cdt+cpt, max_children_time)
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
    estimated_end = design_time + parents_design_time + physical_time + max_children_time
    status = NOT_YET if i.external[p][e] else NO 
    item_id = graph.add_item(p, e, head, i.external[p][e], status, i.external_cost[p][e], i.outsourcing_time[p][e], physical_time, design_time, len(ancestors), len(childrens), parents_physical_time, children_design_time, parents_design_time, estimated_end)
    op_start = parents_design_time
    start, end = i.get_operations_idx(p,e)
    for o in range(start, end):
        succs = end-(o+1)
        minutes = not (i.in_hours[p][o] or i.in_days[p][o])
        operation_time = i.operation_time(p,o)
        op_id = graph.add_operation(p, o, i.is_design[p][o], i.simultaneous[p][o], minutes, i.in_hours[p][o], i.in_days[p][o], succs, childrens_physical_operations + succs, operation_time, i.required_rt(p,o), status, op_start, op_start+operation_time)
        graph.add_operation_assembly(item_id, op_id)
        for r in i.required_resources(p,o):
            if i.finite_capacity[r]:
                graph.add_need_for_resources(op_id, graph.resource_i2g[r], [0, i.execution_time[r][p][o], i.execution_time[r][p][o], op_start, op_start+i.execution_time[r][p][o]])
            else:
                graph.add_need_for_materials(op_id, graph.material_i2g[r], [0, op_start, i.quantity_needed[r][p][o]])
    for children in i.get_children(p, e, True):
        graph, child_id, estimated_end_child = build_item(i, graph, p, children, head=False)
        graph.add_item_assembly(item_id, child_id)
        estimated_end = max(estimated_end, estimated_end_child)
    return graph, item_id, estimated_end

def build_precedence(i: Instance, graph: GraphInstance):
    for p in range(i.get_nb_projects()):
        for e in range(i.E_size[p]):
            start, end = i.get_operations_idx(p, e)
            parent_last_design = []
            parent_first_physical = []
            parent = i.get_direct_parent(p, e)
            if parent > -1:
                parent_last_design = i.last_design_operations(p, parent)
                parent_first_physical = i.first_physical_operations(p, parent)
            for o in range(start, end):
                o_id = graph.operations_i2g[p][o]
                preds = i.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=False, preds=True)
                for pred in preds:
                    pred_id = graph.operations_i2g[p][pred]
                    graph.add_precedence(pred_id, o_id)
                for pld in parent_last_design:
                    pred_id = graph.operations_i2g[p][pld]
                    graph.add_precedence(pred_id, o_id)
                for pfp in parent_first_physical:
                    succ_id = graph.operations_i2g[p][pfp]
                    graph.add_precedence(o_id, succ_id)
    return graph

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
    graph.resources_i2g = graph.build_i2g_1D(graph.resources_g2i)
    graph.operations_i2g = graph.build_i2g_1D(graph.materials_g2i)
    for p in range(i.get_nb_projects()):
        head = i.project_head(p)
        graph, _, lower_bound = build_item(i, graph, p, head, head=True)
    graph.operations_i2g = graph.build_i2g_2D(graph.operations_g2i)
    graph.items_i2g = graph.build_i2g_2D(graph.items_g2i)
    graph = build_precedence(i, graph)
    return graph, lower_bound

# =====================================================
# =*= EXECUTE ONE INSTANCE =*=
# =====================================================

def obective_value(cmax, cost, cmax_weight):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def reccursive_outourcing_actions(instance: Instance, graph: GraphInstance, item):
    actions = []
    status = graph.item(item, 'outsourced')
    if graph.item(item, 'external') == YES and status == NOT_YET:
        actions.extend([(item, YES), (item, NO)])
    elif graph.item(item, 'external') == NO or status == NO:
        for child in graph.get_direct_children(instance, item):
            actions.extend(reccursive_outourcing_actions(instance, graph, child))
    return actions

def check_time(instance: Instance, current_time, hours=True, days=True):
    must_be = 60*instance.H if days else 60 if hours else 1
    return current_time % must_be == 0

def check_scheduling_action(instance: Instance, graph: GraphInstance, operation_id, p, e, start, end, o, required_types_of_resources, required_types_of_materials, res_by_types, current_time):
    actions = []
    can_be_executed = False
    if check_time(instance, current_time, instance.in_hours[p][o], instance.in_days[p][o]): 
        preds = instance.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=False, preds=True)
        still_has_preds_to_execute = False
        for pred in preds:
            if graph.operation(graph.operations_i2g[p][pred], 'remaining_resources') > 0:
                still_has_preds_to_execute = True
                break
        if not still_has_preds_to_execute:
            sync_available = True
            sync_actions = []
            if not instance.simultaneous[p][o]:
                can_be_executed = True
            for rt in required_types_of_resources[p][o]:
                for r in res_by_types[rt]:
                    res_id = graph.resources_i2g[r]
                    if not instance.simultaneous[p][o] and graph.resource(res_id, 'available_time') <= current_time:
                        actions.append((operation_id, res_id))
                    if instance.simultaneous[p][o]:
                        if graph.resource(res_id, 'available_time') <= current_time:
                            sync_actions.append((operation_id, res_id))
                        else:
                            sync_available = False
                            break
            if instance.simultaneous[p][o] and sync_available:
                for rt in required_types_of_materials[p][o]:
                    for m in res_by_types[rt]:
                        mat_id = graph.materials_i2g[m]
                        if instance.purchase_time[m] > current_time and graph.material(mat_id, 'quantity') < instance.quantity_needed[m][p][o]:
                            sync_available = False
                            break
                if sync_available:
                    actions.extend(sync_actions)
                    can_be_executed = True
    return actions, can_be_executed

def reccursive_scheduling_actions(instance: Instance, graph: GraphInstance, item, required_types_of_resources, required_types_of_materials, res_by_types, current_time):
    actions = []
    operations = []
    if graph.item(item, 'external') == NO or graph.item(item, 'outsourced') == NO:
        p, e = graph.items_g2i[item]
        start, end = instance.get_operations_idx(p, e)
        start_design = start
        for o in range(start, end):
            if instance.is_design[p][o]:
                start_design = o
                operation_id = graph.operations_i2g[p][o]
                actions_o, can_be_executed = check_scheduling_action(instance, graph, operation_id, p, e, start, end, o, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
                actions.extend(actions_o)
                if can_be_executed:
                    operations.append(operation_id)
        if not actions: # no design operations left
            for child_i in instance.get_children(p, e, direct=True): 
                child = graph.items_i2g[p][child_i]
                p_actions, p_operations = reccursive_scheduling_actions(instance, graph, child, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
                actions.extend(p_actions)
                actions.extend(p_operations)
        if not actions: # no children to execute
            for o in range(start_design+1, end):
                operation_id = graph.operations_i2g[p][o]
                actions_o, can_be_executed = check_scheduling_action(instance, graph, p, e, start, end, o, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
                actions.extend(actions_o)
                if can_be_executed:
                    operations.append(operation_id)
    return actions, operations

def get_outourcing_actions(instance: Instance, graph: GraphInstance):
    actions = []
    for project_head in graph.project_heads:
        actions.extend(reccursive_outourcing_actions(instance, graph, project_head))
    return actions

def get_scheduling_actions(instance: Instance, graph: GraphInstance, required_types_of_resources, required_types_of_materials, res_by_types, current_time):
    actions = []
    operations = []
    for project_head in graph.project_heads:
        p_actions, p_operations = reccursive_scheduling_actions(instance, graph, project_head, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
        actions.extend(p_actions)
        operations.extend(p_operations)
    return actions, operations

def get_material_use_actions(instance: Instance, graph: GraphInstance, operations, required_types_of_materials, res_by_types, current_time):
    actions = []
    for operation_id in operations:
        p, o = graph.operations_g2i(operation_id)
        for rt in required_types_of_materials[p][o]:
            for m in res_by_types[rt]:
                mat_id = graph.materials_i2g[m]
                if instance.purchase_time[m] <= current_time or graph.material(mat_id, 'quantity') >= instance.quantity_needed[m][p][o]: 
                    actions.append((operation_id, mat_id))
    return actions

def get_feasible_actions(instance: Instance, graph: GraphInstance, required_types_of_resources, required_types_of_materials, res_by_types, current_time, check_for_outsourcing=True):
    actions = get_outourcing_actions(instance, graph) if check_for_outsourcing else []
    type = OUTSOURCING
    if not actions:
        actions, operations = get_scheduling_actions(instance, graph, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
        type = SCHEDULING
    if not actions and len(operations)>0: 
        actions = get_material_use_actions(instance, graph, operations, required_types_of_materials, res_by_types, current_time)
        type = MATERIAL_USE
    return actions, type

def build_required_resources(i: Instance):
    required_types_of_resources = [[] for _ in range(i.get_nb_projects())]
    required_types_of_materials = [[] for _ in range(i.get_nb_projects())]
    res_by_types = [[] for _ in range(i.nb_resource_types)]
    for r in range(i.nb_resources):
        res_by_types[i.get_resource_familly(r)].append(r)
    for p in range(i.get_nb_projects()):
        nb_ops = i.O_size[p]
        required_types_of_resources[p] = [[] for _ in range(nb_ops)]
        required_types_of_materials[p] = [[] for _ in range(nb_ops)]
        for o in range(nb_ops):
            for rt in i.required_rt(p, o):
                if i.finite_capacity[i.resources_by_type(rt)[0]]:
                    required_types_of_resources[p][o].append(rt)
                else:
                    required_types_of_materials[p][o].append(rt)
    return required_types_of_resources, required_types_of_materials, res_by_types

def solve_one(instance: Instance, shared_GNN: L1_EMBEDDING_GNN, outsourcing_actor: L1_ACTOR_CRITIC_GNN, scheduling_actor: L1_ACTOR_CRITIC_GNN, material_actor: L1_ACTOR_CRITIC_GNN, path, train = False, save=False):
    graph, current_cmax = translate(instance)
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    current_time = 0
    current_cost = 0
    actions, actions_type = get_feasible_actions(instance, graph, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
    return current_cmax, current_cost

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
        solve_one(instance, shared_GNN, outsourcing_actor, scheduling_actor, material_actor, SOLUTION_PATH, train=False, save=True)
    print("===* END OF FILE *===")