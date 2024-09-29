import argparse
import pickle
from torch.multiprocessing import Pool, set_start_method
import os
from model import Instance, GraphInstance, L1_EmbbedingGNN, L1_MaterialActor, L1_OutousrcingActor, L1_SchedulingActor, ItemFeatures, OperationFeatures, MaterialFeatures, ResourceFeatures, NeedForMaterialFeatures, NeedForResourceFeatures, NO, NOT_YET, YES
from common import load_instance, to_bool, init_several_1D, search_object_by_id
import torch
torch.autograd.set_detect_anomaly(True)
import random
import multiprocessing
import pandas as pd
import time as systime

PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
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
BASIC_PATH = "./"

# =====================================================
# =*= MODEL LOADING METHOD =*=
# =====================================================

def save_models(agents):
    torch.save(agents[0].shared_embedding_layers.state_dict(), BASIC_PATH+'models/gnn_weights.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), BASIC_PATH+'models/'+name+'_weights.pth')

def load_trained_models():
    shared_GNN = L1_EmbbedingGNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(BASIC_PATH+'models/gnn_weights.pth'))
    outsourcing_actor = L1_OutousrcingActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    scheduling_actor = L1_SchedulingActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    material_actor = L1_MaterialActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    outsourcing_actor.load_state_dict(torch.load(BASIC_PATH+'models/outsourcing_weights.pth'))
    scheduling_actor.load_state_dict(torch.load(BASIC_PATH+'models/scheduling_weights.pth'))
    material_actor.load_state_dict(torch.load(BASIC_PATH+'models/material_weights.pth'))
    return [(outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')]

def init_new_models():
    shared_GNN = L1_EmbbedingGNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    outsourcing_actor = L1_OutousrcingActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    scheduling_actor = L1_SchedulingActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    material_actor = L1_MaterialActor(shared_GNN, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    return [(outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')]

# =====================================================
# =*= TRANSLATE INSTANCE TO GRAPH =*=
# =====================================================

def build_item(i: Instance, graph: GraphInstance, p, e, head):
    design_time, physical_time = i.item_processing_time(p, e)
    children_time = 0
    childrens = i.get_children(p, e, direct=False)
    childrens_physical_operations = 0
    childrens_design_operations = 0
    for children in childrens:
        cdt, cpt = i.item_processing_time(p, children)
        children_time += (cdt+cpt)
        start_c, end_c = i.get_operations_idx(p, children)
        for child_op in range(start_c, end_c):
            if not i.is_design[p][child_op]:
                childrens_physical_operations += 1
            else:
                childrens_design_operations +=1
    parents_design_time = 0
    parents_physical_time = 0
    ancestors = i.get_ancestors(p, e)
    for ancestor in ancestors:
            adt, apt = i.item_processing_time(p, ancestor)
            parents_physical_time += apt
            parents_design_time += adt
    estimated_end = design_time + parents_design_time + physical_time + children_time
    item_id = graph.add_item(p, e, ItemFeatures(
        head = head, 
        external = i.external[p][e],
        outsourced = NOT_YET if i.external[p][e] else NO,
        outsourcing_cost = i.external_cost[p][e],
        outsourcing_time = i.outsourcing_time[p][e],
        remaining_physical_time = physical_time,
        remaining_design_time = design_time,
        parents = len(ancestors),
        children = len(childrens),
        parents_physical_time = parents_physical_time,
        children_time = children_time,
        start_time = parents_design_time,
        end_time = estimated_end,
        is_possible = YES if head else NOT_YET))
    op_start = parents_design_time
    start, end = i.get_operations_idx(p,e)
    for o in range(start, end):
        if o > start:
            op_start = op_start+operation_time
        succs = end-(o+1)
        operation_time = i.operation_time(p,o)
        required_res = 0
        required_mat = 0
        for rt in i.required_rt(p, o):
            resources_of_rt = i.resources_by_type(rt)
            if resources_of_rt:
                if i.finite_capacity[resources_of_rt[0]]:
                    required_res += 1
                else:
                    required_mat += 1
        op_id = graph.add_operation(p, o, OperationFeatures(
            design = i.is_design[p][o],
            sync = i.simultaneous[p][o],
            timescale_hours = i.in_hours[p][o],
            timescale_days = i.in_days[p][o],
            direct_successors = succs,
            total_successors = childrens_physical_operations + succs + (childrens_design_operations if i.is_design[p][o] else 0),
            remaining_time = operation_time,
            remaining_resources = required_res,
            remaining_materials = required_mat,
            available_time = op_start,
            end_time = op_start+operation_time,
            is_possible = YES if (head and (o == start)) else NOT_YET))
        graph.add_operation_assembly(item_id, op_id)
        for r in i.required_resources(p,o):
            if i.finite_capacity[r]:
                graph.add_need_for_resources(op_id, graph.resources_i2g[r], NeedForResourceFeatures(
                    status = NOT_YET,
                    basic_processing_time = i.execution_time[r][p][o],
                    current_processing_time = i.execution_time[r][p][o],
                    start_time = op_start,
                    end_time = op_start+i.execution_time[r][p][o]))
            else:
                graph.add_need_for_materials(op_id, graph.materials_i2g[r], NeedForMaterialFeatures(
                    status = NOT_YET,
                    execution_time = op_start,
                    quantity_needed = i.quantity_needed[r][p][o]))
    for children in i.get_children(p, e, True):
        graph, child_id, estimated_end_child = build_item(i, graph, p, children, head=False)
        graph.add_item_assembly(item_id, child_id)
        estimated_end = max(estimated_end, estimated_end_child)
    return graph, item_id, estimated_end

def build_precedence(i: Instance, graph: GraphInstance):
    for p in i.loop_projects():
        for e in i.loop_items(p):
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
    for rt in range(i.nb_resource_types):
        resources = i.resources_by_type(rt)
        operations = i.operations_by_resource_type(rt)
        res_idx = []
        for r in resources:
            if i.finite_capacity[r]:
                res_id = graph.add_resource(r, i.nb_settings, ResourceFeatures(
                    utilization_ratio = 0,
                    available_time = 0,
                    executed_operations = 0,
                    remaining_operations = len(operations),
                    similar_resources = len(resources)))
                for other_id in res_idx:
                    graph.add_same_types(other_id, res_id)
                res_idx.append(res_id)
            else:
                remaining_quantity_needed = 0
                for p, o in operations:
                    remaining_quantity_needed += i.quantity_needed[r][p][o]
                graph.add_material(r, MaterialFeatures(
                    remaining_init_quantity = i.init_quantity[r], 
                    arrival_time = i.purchase_time[r], 
                    remaining_demand = remaining_quantity_needed))
    graph.resources_i2g = graph.build_i2g_1D(graph.resources_g2i, i.nb_resources)
    graph.materials_i2g = graph.build_i2g_1D(graph.materials_g2i, i.nb_resources)
    for p in i.loop_projects():
        head = i.project_head(p)
        graph, _, lower_bound = build_item(i, graph, p, head, head=True)
    graph.operations_i2g = graph.build_i2g_2D(graph.operations_g2i)
    graph.items_i2g = graph.build_i2g_2D(graph.items_g2i)
    graph.add_dummy_item()
    graph = build_precedence(i, graph)
    return graph, lower_bound

# =====================================================
# =*= SEARCH FOR FEASIBLE ACTIONS =*=
# =====================================================

def reccursive_outourcing_actions(instance: Instance, graph: GraphInstance, item_id):
    actions = []
    external = graph.item(item_id, 'external')
    decision_made = graph.item(item_id, 'outsourced')
    available = graph.item(item_id, 'is_possible')
    if available==YES:
        if external==YES and decision_made==NOT_YET:
            p, e = graph.items_g2i[item_id]
            need_to_be_outsourced = False
            for o in instance.loop_item_operations(p,e):
                for rt in instance.required_rt(p, o):
                    if not instance.resources_by_type(rt):
                        need_to_be_outsourced = True
                        break
                if need_to_be_outsourced:
                    break
            if need_to_be_outsourced:
                actions.append((item_id, YES))
            else:
                actions.extend([(item_id, YES), (item_id, NO)])
        elif external==NO or decision_made==NO:
            for child in graph.get_direct_children(instance, item_id):
                actions.extend(reccursive_outourcing_actions(instance, graph, child))
    return actions

def get_outourcing_actions(instance: Instance, graph: GraphInstance):
    actions = []
    for project_head in graph.project_heads:
        actions.extend(reccursive_outourcing_actions(instance, graph, project_head))
    return actions

def get_scheduling_and_material_use_actions(instance: Instance, graph: GraphInstance, operations, required_types_of_resources, required_types_of_materials, res_by_types, current_time, debug_print):
    scheduling_actions = []
    material_use_actions = []
    ops_to_test = operations if operations else graph.loop_operations()
    for operation_id in ops_to_test:
        p, o = graph.operations_g2i[operation_id]
        e = instance.get_item_of_operation(p, o)
        item_id = graph.items_i2g[p][e]
        timescale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
        if graph.item(item_id, 'is_possible')==YES \
                and (graph.item(item_id, 'external')==NO or graph.item(item_id, 'outsourced')==NO) \
                and graph.operation(operation_id, 'is_possible') == YES \
                and graph.operation(operation_id, 'available_time') <= current_time \
                and current_time % timescale == 0:
            can_search_for_material_use = True
            if graph.operation(operation_id, 'remaining_resources')>0: # 1. Try for scheduling (and check for sync)
                sync_available = True
                sync_actions = []
                for rt in required_types_of_resources[p][o]:
                    for r in res_by_types[rt]:
                        res_id = graph.resources_i2g[r]
                        if not instance.simultaneous[p][o] and graph.resource(res_id, 'available_time') <= current_time:
                            scheduling_actions.append((operation_id, res_id))
                        if instance.simultaneous[p][o]:
                            if graph.resource(res_id, 'available_time') <= current_time:
                                sync_actions.append((operation_id, res_id))
                            else:
                                debug_print(f"\t Sync impossible at time {current_time} for Operation {operation_id} -> ({p}, {o}) due to resource {res_id}/{r}...")
                                sync_available = False
                                break
                    if not sync_available:
                        break
                if instance.simultaneous[p][o] and sync_available:
                    for rt in required_types_of_materials[p][o]:
                        for m in res_by_types[rt]:
                            mat_id = graph.materials_i2g[m]
                            if instance.purchase_time[m] > current_time and graph.material(mat_id, 'remaining_init_quantity') < instance.quantity_needed[m][p][o]:
                                debug_print(f"\t Sync impossible at time {current_time} for Operation {operation_id} -> ({p}, {o}) due to material {mat_id}/{m}...")
                                sync_available = False
                                break
                        if not sync_available:
                            break
                if sync_available:
                    scheduling_actions.extend(sync_actions)
                can_search_for_material_use = sync_available
            elif can_search_for_material_use and graph.operation(operation_id, 'remaining_materials')>0: # 2. Try for material use
                for rt in required_types_of_materials[p][o]:
                    for m in res_by_types[rt]:
                        mat_id = graph.materials_i2g[m]
                        if instance.purchase_time[m] <= current_time or graph.material(mat_id, 'remaining_init_quantity') >= instance.quantity_needed[m][p][o]: 
                            material_use_actions.append((operation_id, mat_id))
    return scheduling_actions, material_use_actions

def get_feasible_actions(instance: Instance, graph: GraphInstance, operations, required_types_of_resources, required_types_of_materials, res_by_types, current_time, debug_print):
    actions = get_outourcing_actions(instance, graph)
    type = OUTSOURCING
    if not actions:
        scheduling_actions, material_use_actions = get_scheduling_and_material_use_actions(instance, graph, operations,required_types_of_resources, required_types_of_materials, res_by_types, current_time, debug_print)
        if scheduling_actions:
            actions = scheduling_actions
            type = SCHEDULING
        elif material_use_actions:
            actions = material_use_actions
            type = MATERIAL_USE
    return actions, type

# =====================================================
# =*= EXECUTE ONE INSTANCE =*=
# =====================================================

def objective_value(cmax, cost, cmax_weight):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def build_required_resources(i: Instance):
    required_types_of_resources = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    required_types_of_materials = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    res_by_types = [[] for _ in range(i.nb_resource_types)]
    for r in range(i.nb_resources):
        res_by_types[i.get_resource_familly(r)].append(r)
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for rt in i.required_rt(p, o):
                resources_of_rt = i.resources_by_type(rt)
                if resources_of_rt:
                    if i.finite_capacity[resources_of_rt[0]]:
                        required_types_of_resources[p][o].append(rt)
                    else:
                        required_types_of_materials[p][o].append(rt)
                else:
                    print(f'\t -> Operation ({p},{o}) requires type ({rt}), which does not have any resources!')
    return required_types_of_resources, required_types_of_materials, res_by_types

def policy(probabilities, greedy=True):
    return torch.argmax(probabilities.view(-1)).item() if greedy else torch.multinomial(probabilities.view(-1), 1).item()

def reward(a, cost_old, cost_new, makespan_old, makespan_new):
    return a * (cost_old - cost_new) + (1-a) * (makespan_old - makespan_new)

def update_processing_time(instance: Instance, graph: GraphInstance, op_id, res_id):
    p, o = graph.operations_g2i[op_id]
    r = graph.resources_g2i[res_id]
    processing_time =  graph.need_for_resource(op_id, res_id, 'basic_processing_time')
    op_setup_time = 0 if (instance.get_operation_type(p, o) == graph.current_operation_type[res_id] or graph.current_operation_type[res_id]<0) else instance.operation_setup[r]
    for d in range(instance.nb_settings):
        dtime = 0 if (graph.current_design_value[res_id][d] == instance.design_value[p][o][d] or graph.current_design_value[res_id][d]<0) else instance.design_setup[r][d] 
        op_setup_time = op_setup_time + dtime
    return processing_time + op_setup_time

def next_possible_time(instance: Instance, current_time, p, o):
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if current_time % scale == 0:
        return current_time
    else:
        return ((current_time // scale) + 1) * scale

def outsource_item(graph: GraphInstance, instance: Instance, item_id, t, enforce_time=False):
    cost = graph.item(item_id, 'outsourcing_cost')
    outsourcing_time = t if enforce_time else max(graph.item(item_id, 'start_time'), t) 
    end_date = outsourcing_time + graph.item(item_id, 'outsourcing_time')
    graph.update_item(item_id, [
        ('outsourced', YES),
        ('is_possible', YES),
        ('remaining_physical_time', 0),
        ('remaining_design_time', 0),
        ('children_time', 0),
        ('start_time', outsourcing_time),
        ('end_time', end_date)])
    p, e = graph.items_g2i[item_id]
    for o in instance.loop_item_operations(p,e):
        op_id = graph.operations_i2g[p][o]
        available_time = next_possible_time(instance, outsourcing_time, p, o)
        graph.update_operation(op_id, [
            ('remaining_resources', 0),
            ('remaining_materials', 0),
            ('remaining_time', 0),
            ('end_time', end_date),
            ('available_time', available_time),
            ('is_possible', YES)]) 
        for r in instance.required_resources(p, o):
            if instance.finite_capacity[r]:
                res_id = graph.resources_i2g[r]
                graph.del_need_for_resource(op_id, res_id)
                graph.inc_resource(res_id, [('remaining_operations', -1)])
            else:
                mat_id = graph.materials_i2g[r]
                quantity_needed = graph.need_for_material(op_id, mat_id, 'quantity_needed')
                graph.del_need_for_material(op_id, mat_id)
                graph.inc_material(mat_id, [('remaining_demand', -1 * quantity_needed)])
    for child in instance.get_children(p, e):
        graph, child_time, child_cost = outsource_item(graph, instance, graph.items_i2g[p][child], outsourcing_time, enforce_time=True)
        cost += child_cost
        end_date = max(t, child_time)
    return graph, end_date, cost

def apply_use_material(graph: GraphInstance, instance: Instance, operation_id, material_id, required_types_of_materials, current_time):
    p, o = graph.operations_g2i[operation_id]
    rt = instance.get_resource_familly(graph.materials_g2i[material_id])
    quantity_needed = graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity = graph.material(material_id, 'remaining_init_quantity')
    waiting_demand = graph.material(material_id, 'remaining_demand') 
    graph.update_need_for_material(operation_id, material_id, [
        ('status', YES), 
        ('execution_time', current_time)
    ])
    graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)
    ])
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', max(current_time, graph.operation(operation_id, 'end_time')))
    ])
    item_id = graph.items_i2g[p][instance.get_item_of_operation(p,o)]
    graph.update_item(item_id, [
        ('end_time', max(current_time, graph.item(item_id, 'end_time'))),
        ('start_time', min(current_time, graph.item(item_id, 'start_time')))
    ])
    required_types_of_materials[p][o].remove(rt)
    return graph, required_types_of_materials

def schedule_operation(graph: GraphInstance, instance: Instance, operation_id, resource_id, required_types_of_resources, utilization, current_time):
    current_processing_time = graph.need_for_resource(operation_id, resource_id, 'current_processing_time')
    operation_end = current_time + current_processing_time
    p, o = graph.operations_g2i[operation_id]
    e = instance.get_item_of_operation(p, o)
    r = graph.resources_g2i[resource_id]
    rt = instance.get_resource_familly(r)
    estimated_processing_time = instance.operation_resource_time(p, o, rt)
    item_id = graph.items_i2g[p][e]
    graph.inc_resource(resource_id, [('executed_operations', 1), ('remaining_operations', -1)])
    graph.update_resource(resource_id, [('available_time', operation_end)])
    graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', current_time),
        ('end_time', operation_end)
    ])
    utilization[resource_id] += current_processing_time
    graph.current_operation_type[resource_id] = instance.get_operation_type(p, o)
    for d in range(instance.nb_settings):
        graph.current_design_value[resource_id][d] == instance.design_value[p][o][d]
    required_types_of_resources[p][o].remove(rt)
    for similar in instance.resources_by_type(rt):
        if similar != r:
            similar_id = graph.resources_i2g[similar]
            graph.inc_resource(similar_id, [('remaining_operations', -1)])
            graph.del_need_for_resource(operation_id, similar_id)
    graph.inc_operation(operation_id, [('remaining_resources', -1), ('remaining_time', -estimated_processing_time)])
    graph.update_operation(operation_id, [('end_time', operation_end)])
    for ancestor in instance.get_ancestors(p, e):
        graph.inc_item(graph.items_i2g[p][ancestor], [
            ('children_time', -estimated_processing_time)
        ])
    if not instance.is_design[p][o]:
        for child in instance.get_children(p, e, direct=False):
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    graph.update_item(item_id, [
        ('start_time', min(current_time, graph.item(item_id, 'start_time'))),
        ('end_time', max(operation_end, graph.item(item_id, 'end_time')))
    ])
    if instance.is_design[p][o]:
        graph.inc_item(item_id, [('remaining_design_time', -estimated_processing_time)])
    else:
        graph.inc_item(item_id, [('remaining_physical_time', -estimated_processing_time)])
    return graph, utilization, required_types_of_resources, operation_end

def try_to_open_next_operations(graph: GraphInstance, instance: Instance, previous_operations, next_operations, operation_id, available_time, debug_print): 
    if graph.is_operation_complete(operation_id):
        p, o = graph.operations_g2i[operation_id]
        e = instance.get_item_of_operation(p, o)
        for next in next_operations[p][o]:
            next_good_to_go = True
            for previous in previous_operations[p][next]:
                if previous != operation_id and not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                    next_good_to_go = False
                    break
            if next_good_to_go:
                next_id = graph.operations_i2g[p][next]
                next_time = next_possible_time(instance, available_time, p, next)
                debug_print(f'Enabling operation ({p},{next}) at time {available_time} -> {next_time}...')
                graph.update_operation(next_id, [
                    ('available_time', next_time),
                    ('is_possible', YES)
                ])
        if instance.is_last_design(p, e, o):
            for child in instance.get_children(p, e, direct=True):
                child_id = graph.items_i2g[p][child]
                debug_print(f'Enabling item {child_id} -> ({p},{child}) for outsourcing...')
                graph.update_item(child_id, [('is_possible', YES), ('start_time', available_time)])
    return graph

def check_completeness(graph: GraphInstance, debug_print):
    for item_id in graph.loop_items():
        outsourced = graph.item(item_id, 'outsourced')
        e = graph.items_g2i[item_id]
        if graph.item(item_id, 'remaining_physical_time')>0:
            debug_print(f"PROBLEM ITEM {e} [outsourced={outsourced}] STILL HAS {graph.item(item_id, 'remaining_physical_time')} OF PHYSICAL TIME TO DO!")
        if graph.item(item_id, 'remaining_design_time')>0:
            debug_print(f"PROBLEM ITEM {e} [outsourced={outsourced}] STILL HAS {graph.item(item_id, 'remaining_design_time')} OF DESIGN TIME TO DO!")  
        if graph.item(item_id, 'outsourced')==NOT_YET:
            debug_print(f"PROBLEM ITEM {e} [outsourced={outsourced}] STILL NO OUTSOURCING DECISIONS!")   
        if graph.item(item_id, 'is_possible')==NOT_YET:
            debug_print(f"PROBLEM ITEM {e} [outsourced={outsourced}] STILL NOT POSSIBLE!")
    for material_id in graph.loop_materials():
        m = graph.materials_g2i[material_id]
        if graph.material(material_id, 'remaining_demand')>0:
            debug_print(f"PROBLEM MATERIAL {m} STILL HAS {graph.material(material_id, 'remaining_demand')} OF REMAINING DEMAND!")
    for resource_id in graph.loop_resources():
        r = graph.resources_g2i[resource_id]
        if graph.resource(resource_id, 'remaining_operations')>0:
            debug_print(f"PROBLEM RESOURCE {r} STILL HAS {graph.resource(resource_id, 'remaining_operations')} OF REMAINING OPERATION!")
    need_for_mat_idx, loop_mat = graph.loop_need_for_material()
    for i in loop_mat:
        operation_id = need_for_mat_idx[0, i]
        material_id = need_for_mat_idx[1, i]
        p, o = graph.operations_g2i[operation_id]
        m = graph.materials_g2i[material_id]
        if graph.need_for_material(operation_id, material_id, 'status') == NOT_YET:
            debug_print(f"PROBLEM NEED OF MATERIAL op=({p},{o}), mat={m} STATUS STILL NO YET!")
    need_for_res_idx, loop_res = graph.loop_need_for_resource()
    for i in loop_res:
        operation_id = need_for_res_idx[0, i]
        resource_id = need_for_res_idx[1, i]
        p, o = graph.operations_g2i[operation_id]
        r = graph.resources_g2i[resource_id]
        if graph.need_for_resource(operation_id, resource_id, 'status') == NOT_YET:
            debug_print(f"PROBLEM NEED OF RESOURCE op=({p},{o}), res={r} STATUS STILL NO YET!")
    for operation_id in graph.loop_operations():
        p, o = graph.operations_g2i[operation_id]
        if graph.operation(operation_id, 'is_possible')==NOT_YET:
            debug_print(f"PROBLEM OPERATION ({p},{o}) STILL NOT POSSIBLE!")
        if graph.operation(operation_id, 'remaining_resources')>0:
            debug_print(f"PROBLEM OPERATION ({p},{o}) STILL {graph.operation(operation_id, 'remaining_resources')} REMAINING RESOURCES!")
        if graph.operation(operation_id, 'remaining_materials')>0:
            debug_print(f"PROBLEM OPERATION ({p},{o}) STILL {graph.operation(operation_id, 'remaining_materials')} REMAINING MATERIALS!")
        if graph.operation(operation_id, 'remaining_time')>0:
            debug_print(f"PROBLEM OPERATION ({p},{o}) STILL {graph.operation(operation_id, 'remaining_time')} REMAINING TIME!")

def solve_one(instance: Instance, agents, path="", train=False, debug_mode=False):
    if debug_mode:
        def debug_print(*args):
            print(*args)
            with open('./log.sh', 'a') as file:
                file.write(*args)
                file.write('\n')
    else:
        def debug_print(*args):
            pass
    debug_print(instance.display())
    start_time = systime.time()
    graph, current_cmax = translate(instance)
    old_cmax = current_cmax
    parents = graph.flatten_parents()
    utilization = [0 for _ in graph.loop_resources()]
    related_items = graph.flatten_related_items()
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    previous_operations, next_operations = instance.build_next_and_previous_operations()
    t = 0
    current_cost = 0
    old_cost = 0
    rewards, values = [torch.Tensor([]) for _ in agents], [torch.Tensor([]) for _ in agents]
    probabilities, states, actions, actions_idx = [[] for _ in agents], [[] for _ in agents], [[] for _ in agents], [[] for _ in agents]
    terminate = False
    operations_to_test = []
    while not terminate:
        poss_actions, actions_type = get_feasible_actions(instance, graph, operations_to_test, required_types_of_resources, required_types_of_materials, res_by_types, t, debug_print)
        debug_print(f"Current possible actions: {poss_actions}")
        if poss_actions:
            if actions_type == SCHEDULING:
                for op_id, res_id in poss_actions:
                    graph.update_need_for_resource(op_id, res_id, [('current_processing_time', update_processing_time(instance, graph, op_id, res_id))])
            probs, state_value = agents[actions_type][AGENT](graph.to_state(), poss_actions, related_items, parents, instance.w_makespan)
            states[actions_type].append(graph.to_state())
            values[actions_type] = torch.cat((values[actions_type], torch.Tensor([state_value.detach()])))
            actions[actions_type].append(poss_actions)
            probabilities[actions_type].append(probs.detach())
            idx = policy(probs, greedy=(not train))
            actions_idx[actions_type].append(idx)
            if actions_type == OUTSOURCING:
                item_id, outsourcing_choice = poss_actions[idx]
                p, e = graph.items_g2i[item_id]
                if outsourcing_choice == YES:
                    graph, end_date, local_price = outsource_item(graph, instance, item_id, t, enforce_time=False)
                    debug_print(f"Outsourcing item {item_id} -> ({p},{e})...")
                    approximate_d_time, approximate_p_time = instance.item_processing_time(p, e)
                    for ancestor in instance.get_ancestors(p, e):
                        ancestor_id = graph.items_i2g[p][ancestor]
                        max_ancestor_end = max(end_date, graph.item(ancestor_id, 'end_time'))
                        graph.update_item(ancestor_id, [
                            ('children_time', graph.item(ancestor_id, 'children_time')-(approximate_d_time+approximate_p_time)),
                            ('end_time', max_ancestor_end)
                        ])
                    for o in instance.first_physical_operations(p, instance.get_direct_parent(p, e)):
                        next_good_to_go = True
                        for previous in previous_operations[p][o]:
                            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                                next_good_to_go = False
                                break
                        if next_good_to_go:
                            op_id = graph.operations_i2g[p][o]
                            available_time = next_possible_time(instance, end_date, p, o)
                            graph.update_operation(op_id, [('is_possible', YES), ('available_time', available_time)])
                    current_cost += local_price  
                    current_cmax = max(current_cmax, max_ancestor_end)
                else:
                    debug_print(f"Producing item {item_id} -> ({p},{e}) locally...")
                    graph.update_item(item_id, [('outsourced', NO)])
            elif actions_type == SCHEDULING:
                operation_id, resource_id = poss_actions[idx]    
                p, o = graph.operations_g2i[operation_id]
                debug_print(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {t}...")     
                graph, utilization, required_types_of_resources, operation_end = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, utilization, t)
                debug_print(f"End of scheduling at time {operation_end}...")
                if instance.simultaneous[p][o]:
                    for rt in instance.required_rt(p, o):
                        if rt != instance.get_resource_familly(graph.resources_g2i[resource_id]):
                            for r in instance.resources_by_type(rt):
                                if instance.finite_capacity[r]:
                                    other_resource_id = graph.resources_i2g[r]
                                    if graph.resource(other_resource_id, 'available_time') <= t:
                                        graph, utilization, required_types_of_resources, op_end = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, utilization, t)
                                        operation_end = max(operation_end, op_end)
                                        break
                                else:
                                    graph, required_types_of_materials = apply_use_material(graph, instance, operation_id, graph.materials_i2g[r], required_types_of_materials, t)
                                    break
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, operation_end, debug_print)
                current_cmax = max(current_cmax, operation_end)
            else:
                operation_id, material_id = poss_actions[idx]
                p, o = graph.operations_g2i[operation_id]
                debug_print(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]}...")  
                graph, required_types_of_materials = apply_use_material(graph, instance, operation_id, material_id, required_types_of_materials, t)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, t, debug_print)
                current_cmax = max(current_cmax, t)
            reward(instance.w_makespan, old_cost, current_cost, old_cmax, current_cmax)
            old_cost = current_cost
            old_cmax = current_cmax
        else: # NO OPERATIONS LEFT AT TIME T SEARCH FOR NEXT TIME
            next_date = -1
            operations_to_test = []
            for material_id in graph.loop_materials():
                arrival_time = graph.material(material_id, 'arrival_time')
                if arrival_time>t and (arrival_time<next_date or next_date<0):
                    debug_print(f"\t --> New quantity of material {material_id} is available at time {arrival_time} [t={t}]")
                    next_date = arrival_time
            for resource_id in graph.loop_resources():
                available_time = graph.resource(resource_id, 'available_time')
                debug_print(f"\t --> Machine {resource_id} is available at time {available_time} [t={t}]")
                if available_time>t and (available_time<next_date or next_date<0):
                    next_date = available_time
            for operation_id in graph.loop_operations():
                if graph.operation(operation_id, 'is_possible') == YES and (graph.operation(operation_id, 'remaining_resources')>0 or graph.operation(operation_id, 'remaining_materials')>0):
                    available_time = graph.operation(operation_id, 'available_time')
                    if available_time <= t:
                        p, o = graph.operations_g2i[operation_id]
                        available_time = next_possible_time(instance, t, p, o)
                    if graph.operation(operation_id, 'remaining_resources')>0:
                        debug_print(f"\t --> operation {operation_id} can be scheduled at time {available_time} [t={t}]")
                    elif graph.operation(operation_id, 'remaining_materials')>0:
                        debug_print(f"\t --> operation {operation_id} can use material at time {available_time} [t={t}]")
                    if available_time>t: 
                        operations_to_test.append(operation_id)
                        if available_time<next_date or next_date<0:
                            next_date = available_time
            if next_date>=0:
                t = next_date
                if t > 0:
                    for res_id in graph.loop_resources():
                        graph.update_resource(res_id, [('utilization_ratio', utilization[res_id] / t)])
                debug_print(f"New current time t={t}...")
            else:
                debug_print("End of solving stage!")
                if debug_mode:
                    check_completeness(graph, debug_print)
                terminate = True
    if train:
        return rewards, values, probabilities, states, actions, actions_idx, [instance.id for _ in rewards], related_items, parents
    else:
        solutions_df = pd.DataFrame({
            'index': [instance.id],
            'value': [objective_value(current_cmax, current_cost, instance.w_makespan)/100], 
            'computing_time': [systime.time()-start_time]
        })
        print(solutions_df)
        solutions_df.to_csv(path, index=False)
        return current_cmax, current_cost 

# ====================================================================
# =*= PPO TRAINING PROCESS FOR MULTI-AGENTS WITH SHARED PARAMETERS =*=
# ====================================================================

def search_instance(instances, id) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(debug_mode):
    instances = [] 
    for size in PROBLEM_SIZES if not debug_mode else ['s', 'm']:
        problems = []
        path = BASIC_PATH+'instances/train/'+size+'/'
        for i in os.listdir(path):
            if i.endswith('.pkl'):
                file_path = os.path.join(path, i)
                with open(file_path, 'rb') as file:
                    problems.append(pickle.load(file))
        instances.extend(problems)
    print(f"End of loading {len(instances)} instances!")
    return instances

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

def PPO_loss(instances, agent, old_probs, states, actions, actions_idx, advantages, old_values, returns, instances_idx, all_related_items, all_parents, e=PPO_CONF['clip_ratio']):
    new_log_probs = torch.Tensor([])
    old_log_probs = torch.Tensor([])
    entropies = torch.Tensor([])
    id = -1
    instance = None
    related_items = []
    parents = []
    for i in range(len(states)):
        if instances_idx[id] != id:
            id = instances_idx[id]
            instance = search_instance(instances, id)
            related_items = search_object_by_id(all_related_items, id)['related_item']
            parents = search_object_by_id(all_parents, id)['parent']
        p,_ = agent(states[i], actions[i], related_items, parents, instance.w_makespan)
        a = actions_idx[i]
        entropies = torch.cat((entropies, torch.sum(-p*torch.log(p+1e-8), dim=-1)))
        new_log_probs = torch.cat((new_log_probs, torch.log(p[a]+1e-8)))
        old_log_probs = torch.cat((old_log_probs, torch.log(old_probs[i][a]+1e-8)))
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-e, 1+e)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = torch.mean(torch.stack([(V_old - r) ** 2 for V_old, r in zip(old_values, returns)]))
    entropy_loss = torch.mean(entropies)
    print(f"\t\t value loss - {value_loss}")
    print(f"\t\t policy loss - {policy_loss}") 
    print(f"\t\t entropy loss - {entropy_loss}") 
    return (PPO_CONF['policy_loss']*policy_loss) + (PPO_CONF['value_loss']*value_loss) - (entropy_loss*PPO_CONF['entropy'])

def PPO_optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

def async_solve_one(init_args):
    agents, instance, debug_mode = init_args
    print(f"\t start solving instance: {instance.id}...")
    result = solve_one(instance, agents, train=True, debug_mode=debug_mode)
    print(f"\t end solving instance: {instance.id}!")
    return result

def async_solve_batch(agents, batch, num_processes, train, epochs, optimizers, debug):
    all_probabilities, all_states, all_actions, all_actions_idx, all_instances_idx = init_several_1D(3, [], 5)
    all_related_items, all_parents = [], []
    with Pool(num_processes) as pool:
        results = pool.map(async_solve_one, [(agents, instance, debug) for instance in batch])
    all_rewards, all_values, probabilities, states, actions, actions_idx, instances_idx, related_items, parents = zip(*results)
    for instance in range(len(batch)):
        all_parents.append({'id': instances_idx[0][0], 'parents': parents[instance]})
        all_related_items.append({'id': instances_idx[0][0], 'related_items': related_items[instance]})
        for agent_id in range(len(agents)):
            all_probabilities[agent_id].extend(probabilities[instance][agent_id])
            all_states[agent_id].extend(states[instance][agent_id])
            all_actions[agent_id].extend(actions[instance][agent_id])
            all_actions_idx[agent_id].extend(actions_idx[instance][agent_id])
            all_instances_idx[agent_id].extend(instances_idx[instance][agent_id])
    all_returns = [[ri for r in agent_rewards for ri in calculate_returns(r)] for agent_rewards in all_rewards]
    advantages = []
    flattened_values = []
    for agent_id, _ in enumerate(agents):
        advantages.append(torch.Tensor([gae for r, v in zip(all_rewards[agent_id], all_values[agent_id]) for gae in generalized_advantage_estimate(r, v)]))
        flattened_values.append([v for vals in all_values[agent_id] for v in vals])
    if train and epochs>0:
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            for agent_id, (agent, name) in enumerate(agent):
                print(f"\t\t Optimizing agent: {name}...")
                loss = PPO_loss(batch, agent, all_probabilities[agent_id], all_states[agent_id], all_actions[agent_id], all_actions_idx[agent_id], advantages[agent_id], flattened_values[agent_id], all_returns[agent_id], all_instances_idx[agent_id], all_related_items, all_parents)
                PPO_optimize(optimizers[agent_id], loss)
    else:
        for agent_id, (agent, name) in enumerate(agents):
            print(f"\t\t Evaluating agent: {name}...")
            loss = PPO_loss(batch, agent, all_probabilities[agent_id], all_states[agent_id], all_actions[agent_id], all_actions_idx[agent_id], advantages[agent_id], flattened_values[agent_id], all_returns[agent_id], all_instances_idx[agent_id], all_related_items, all_parents)
            print(f'\t Average Loss = {loss:.4f}')

def train(instances, agents, iterations, batch_size, epochs, validation_rate, debug_mode=False):
    if debug_mode:
        def debug_print(*args):
            print(*args)
            with open('./log.sh', 'a') as file:
                file.write(*args)
                file.write('\n')
    else:
        def debug_print(*args):
            pass
    optimizers = [torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE) for agent,_ in agents]
    for agent,_ in agents:
        agent.train()
        if DEVICE == "cuda":
            agent.to(DEVICE)
    random.shuffle(instances)
    num_val = int(len(instances) * PPO_CONF['validation_ratio'])
    train_instances, val_instances = instances[num_val:], instances[:num_val]
    num_processes = multiprocessing.cpu_count() if PARRALLEL else 1
    print(f"Start training models with PPO running on {num_processes} TPUs in parallel...")
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % PPO_CONF['switch_batch'] == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch = random.sample(train_instances, batch_size)
        async_solve_batch(agents, current_batch, num_processes, train=True, epochs=epochs, optimizers=optimizers, debug=debug_mode)
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            with torch.no_grad():
                async_solve_batch(agents, val_instances, num_processes, train=False, epochs=-1, optimizers=[], debug=debug_mode)
            for agent,_ in agents:
                agent.train()
    save_models(agents)
    print("<======***--| END OF TRAINING |--***======>")

# =====================================================
# =*= MAIN CODE =*=
# =====================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    parser.add_argument("--path", help="Saving path on the server", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    debug_mode = False
    if args.mode == 'test':
        PPO_CONF['train_iterations'] = 10
        PPO_CONF['batch_size'] = 3
        PARRALLEL = False
        debug_mode = True
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    BASIC_PATH = args.path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device: {DEVICE}...")
    if to_bool(args.train):
        '''
            Test training mode with: bash _env.sh
            python gns_solver.py --train=true --mode=test --path=./
        '''
        instances = load_training_dataset(debug_mode=debug_mode)
        train(instances, init_new_models(), iterations=PPO_CONF['train_iterations'], batch_size=PPO_CONF['batch_size'], epochs=PPO_CONF['opt_epochs'], validation_rate=PPO_CONF['validation_rate'], debug_mode=debug_mode)
    else:
        '''
            Test inference mode with: bash _env.sh
            python gns_solver.py --size=s --id=151 --train=false --mode=test --path=./
        '''
        print(f"SOLVE TARGET INSTANCE {args.size}_{args.id}...")
        INSTANCE_PATH = BASIC_PATH+'instances/test/'+args.size+'/instance_'+args.id+'.pkl'
        SOLUTION_PATH = BASIC_PATH+'instances/test/'+args.size+'/solution_gns_'+args.id+'.csv'
        instance: Instance = load_instance(INSTANCE_PATH)
        agents = init_new_models() if args.mode == 'test' else load_trained_models() 
        solve_one(instance, agents, SOLUTION_PATH, train=False, debug_mode=debug_mode)
    print("===* END OF FILE *===")