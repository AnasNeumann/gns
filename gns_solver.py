import argparse
from model.instance import Instance
from model.graph import GraphInstance, NO, NOT_YET, YES
from model.gnn import L1_EmbbedingGNN, L1_MaterialActor, L1_OutousrcingActor, L1_SchedulingActor, L1_CommonCritic
from model.solution import HeuristicSolution
from tools.common import load_instance, to_bool, directory
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
import time as systime
from typing import Callable
from torch import Tensor
from torch.nn import Module
from translators.instance2graph_translator import translate
from translators.graph2solution_translator import translate_solution
from debug.debug_gns import check_completeness, debug_printer
from gns_ppo_trainer import reward, PPO_pre_train, load_training_dataset, PPO_fine_tuning
from model.agent import MultiAgent_OneInstance
import pickle

# =====================================================
# =*= 1st MAIN FILE OF THE PROJECT: GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

DEBUG_PRINT: callable = None
LEARNING_RATE = 2e-4
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
GNN_CONF = {
    'resource_and_material_embedding_size': 16,
    'operation_and_item_embedding_size': 24,
    'nb_layers': 2,
    'embedding_hidden_channels': 128,
    'value_hidden_channels': 256,
    'actor_hidden_channels': 256}

# =====================================================
# =*= I. SEARCH FOR FEASIBLE ACTIONS =*=
# =====================================================

def reccursive_outourcing_actions(instance: Instance, graph: GraphInstance, item_id: int):
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

def get_scheduling_and_material_use_actions(instance: Instance, graph: GraphInstance, operations: list[int], required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], res_by_types: list[list[int]], current_time: int):
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
                                DEBUG_PRINT(f"\t Sync impossible at time {current_time} for Operation {operation_id} -> ({p}, {o}) due to resource {res_id}/{r}...")
                                sync_available = False
                                break
                    if not sync_available:
                        break
                if instance.simultaneous[p][o] and sync_available:
                    for rt in required_types_of_materials[p][o]:
                        for m in res_by_types[rt]:
                            mat_id = graph.materials_i2g[m]
                            if instance.purchase_time[m] > current_time and graph.material(mat_id, 'remaining_init_quantity') < instance.quantity_needed[m][p][o]:
                                DEBUG_PRINT(f"\t Sync impossible at time {current_time} for Operation {operation_id} -> ({p}, {o}) due to material {mat_id}/{m}...")
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

def get_feasible_actions(instance: Instance, graph: GraphInstance, operations: list[int], required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], res_by_types: list[list[int]], current_time: int):
    actions = get_outourcing_actions(instance, graph)
    type = OUTSOURCING
    if not actions:
        scheduling_actions, material_use_actions = get_scheduling_and_material_use_actions(instance, graph, operations, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
        if scheduling_actions:
            actions = scheduling_actions
            type = SCHEDULING
        elif material_use_actions:
            actions = material_use_actions
            type = MATERIAL_USE
    return actions, type

# =====================================================
# =*= II. APPLY A DECISION MADE =*=
# =====================================================

def shift_one_operation(graph: GraphInstance, instance: Instance, p: int, o: int, shift: int):
    if shift <= 0:
        return graph, 0
    operation_id = graph.operations_i2g[p][o]
    DEBUG_PRINT(f"\t\t >>> Operation ({p},{o}) shifted by {shift} time unit...")
    graph.inc_operation(operation_id, [('available_time', shift), ('end_time', shift)])
    for r in instance.required_resources(p, o):
        if instance.finite_capacity[r]:
            graph.inc_need_for_resource(operation_id, graph.resources_i2g[r], [('start_time', shift), ('end_time', shift)])
        else:
            graph.inc_need_for_material(operation_id, graph.materials_i2g[r], [('execution_time', shift)]) 
    return graph, graph.operation(operation_id, 'end_time')

def shift_next_operations(graph: GraphInstance, instance: Instance, p: int, e: int, o: int, shift: int):
    if shift <= 0:
        return graph, 0
    _, end = instance.get_operations_idx(p, e)
    max_end = 0
    for next in range(o+1, end):
        graph, end_next = shift_one_operation(graph, instance, p, next, shift)
        max_end = max(end_next, max_end)
    return graph, max_end

def shift_children_and_operations(graph: GraphInstance, instance: Instance, p: int, e: int, shift: int):
    if shift <= 0:
        return graph, 0
    max_child_end = 0
    for child in instance.get_children(p, e, direct=False):
        DEBUG_PRINT(f"\t >> Children item ({p},{child}) shifted by {shift} time unit (both P and NP operations)...")
        child_id = graph.items_i2g[p][child]
        graph.inc_item(child_id, [
            ('start_time', shift),
            ('end_time', shift)])
        max_child_end = max(max_child_end, graph.item(child_id, 'end_time'))
        for o in instance.loop_item_operations(p, child):
            graph,_ = shift_one_operation(graph, instance, p, o, shift)
    return graph, max_child_end

def shift_ancestors_physical_operations(graph: GraphInstance, instance: Instance, p: int, e: int, end_time_last_op: int):
    max_ancestor_end = 0
    for ancestor in instance.get_ancestors(p, e):
        min_ancestor_physical_start = -1
        for first_p in instance.first_physical_operations(p, ancestor):
            start = graph.operation(graph.operations_i2g[p][first_p], 'available_time')
            min_ancestor_physical_start = max(min_ancestor_physical_start,start) if min_ancestor_physical_start>0 else start
        shift = max(0, end_time_last_op-min_ancestor_physical_start)
        if shift > 0:
            DEBUG_PRINT(f"\t >> Ancestor item ({p},{ancestor}) shifted by {shift} time unit (only physical operations)...")
            ancestor_id = graph.items_i2g[p][ancestor]
            graph.inc_item(ancestor_id, [('end_time', shift)])
            max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
            for o in instance.loop_item_operations(p, ancestor):
                if not instance.is_design[p][o]:
                    graph,_ = shift_one_operation(graph, instance, p, o, shift)
                    max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
    return graph, max_ancestor_end

def shift_ancestors_and_operations(graph: GraphInstance, instance: Instance, p: int, e: int, shift: int):
    if shift <= 0:
        return graph, 0
    max_ancestor_end = 0
    for ancestor in instance.get_ancestors(p, e):
        DEBUG_PRINT(f"\t >> Ancestor item ({p},{ancestor}) shifted by {shift} time unit (only physical operations)...")
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('end_time', shift)])
        max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
        for o in instance.loop_item_operations(p, ancestor):
            if not instance.is_design[p][o]:
                graph,_ = shift_one_operation(graph, instance, p, o, shift)
                max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
    return graph, max_ancestor_end

def item_local_production(graph: GraphInstance, instance: Instance, item_id: int, p: int, e: int):
    DEBUG_PRINT(f"Producing item {item_id} -> ({p},{e}) locally...")
    old_end = graph.item(item_id, 'end_time')
    max_child_end = 0
    design_mean_time, physical_mean_time = instance.item_processing_time(p, e, total_load=False)
    graph, max_child_end = shift_children_and_operations(graph, instance, p, e, design_mean_time)
    new_end = physical_mean_time + (max_child_end if max_child_end>0 else (graph.item(item_id, 'start_time')+design_mean_time))
    graph, max_ancestors_end = shift_ancestors_and_operations(graph, instance, p, e, shift=new_end-old_end)
    graph.update_item(item_id, [('outsourced', NO), ('end_time', new_end)])
    return graph, max(new_end, max_ancestors_end)

def outsource_item(graph: GraphInstance, instance: Instance, item_id: int, t: int, enforce_time: bool=False):
    """
        Outsource item and children (reccursive down to the leaves)
    """
    cost = graph.item(item_id, 'outsourcing_cost')
    outsourcing_start_time = t if enforce_time else max(graph.item(item_id, 'start_time'), t) 
    p, e = graph.items_g2i[item_id]
    end_date = outsourcing_start_time + instance.outsourcing_time[p][e]
    graph.update_item(item_id, [
        ('outsourced', YES),
        ('is_possible', YES),
        ('remaining_physical_time', 0),
        ('remaining_design_time', 0),
        ('children_time', 0),
        ('start_time', outsourcing_start_time),
        ('end_time', end_date)])
    graph.executed_items += 1
    graph.oustourced_items += 1
    for o in instance.loop_item_operations(p,e):
        op_id = graph.operations_i2g[p][o]
        available_time = next_possible_time(instance, outsourcing_start_time, p, o)
        graph.update_operation(op_id, [
            ('remaining_resources', 0),
            ('remaining_materials', 0),
            ('remaining_time', 0),
            ('end_time', end_date),
            ('available_time', available_time),
            ('is_possible', YES)]) 
        for rt in instance.required_rt(p, o):
            graph.executed_operations += 1
            for r in instance.resources_by_type(rt):
                if instance.finite_capacity[r]:
                    res_id = graph.resources_i2g[r]
                    graph.del_need_for_resource(op_id, res_id)
                    graph.inc_resource(res_id, [('remaining_operations', -1)])
                else:
                    mat_id = graph.materials_i2g[r]
                    quantity_needed = graph.need_for_material(op_id, mat_id, 'quantity_needed')
                    graph.del_need_for_material(op_id, mat_id)
                    graph.inc_material(mat_id, [('remaining_demand', -1 * quantity_needed)])
    for child in instance.get_children(p, e, direct=True):
        graph, child_time, child_cost = outsource_item(graph, instance, graph.items_i2g[p][child], outsourcing_start_time, enforce_time=True)
        cost += child_cost
        end_date = max(t, child_time)
    return graph, end_date, cost

def apply_outsourcing_to_direct_parent(instance: Instance, graph: GraphInstance, current_cmax: int, current_cost: int, previous_operations: list, item_id: int, p: int, e: int, end_date: int, local_price: int):
    """
        Apply an outsourcing decision to the direct parent
    """
    approximate_design_load, approximate_physical_load = instance.item_processing_time(p, e, total_load=True)
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [
            ('children_time', graph.item(ancestor_id, 'children_time')-(approximate_design_load+approximate_physical_load))])
    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, end_date)
    current_cost += local_price
    current_cmax = max(current_cmax, max_ancestors_end)
    DEBUG_PRINT(f"Outsourcing item {item_id} -> ({p},{e}) at time {graph.item(item_id,'start_time')} to {graph.item(item_id,'end_time')} [NEW CMAX = {current_cmax} - NEW COST = {current_cost}]...")
    for o in instance.first_physical_operations(p, instance.get_direct_parent(p, e)):
        next_good_to_go: bool = True
        for previous in previous_operations[p][o]:
            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                DEBUG_PRINT(f"\t >> Cannot open parent' first physical operation ({p},{o}) due to ({p},{previous}) not finished!")
                next_good_to_go = False
                break
        op_id = graph.operations_i2g[p][o]
        open = YES if next_good_to_go else graph.operation(op_id, 'is_possible')
        available_time = max(graph.operation(op_id, 'available_time'), next_possible_time(instance, end_date, p, o))
        DEBUG_PRINT(f"\t >> Moving parent' first physical operation ({p},{o}) to {available_time}!")
        graph.update_operation(op_id, [('is_possible', open)])
        graph.update_operation(op_id, [('available_time', available_time)], maxx=True)
    return graph, current_cmax, current_cost

def apply_use_material(graph: GraphInstance, instance: Instance, operation_id: int, material_id: int, required_types_of_materials:list[list[list[int]]], current_time: int):
    p, o = graph.operations_g2i[operation_id]
    rt = instance.get_resource_familly(graph.materials_g2i[material_id])
    quantity_needed = graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity = graph.material(material_id, 'remaining_init_quantity')
    waiting_demand = graph.material(material_id, 'remaining_demand') 
    graph.update_need_for_material(operation_id, material_id, [
        ('status', YES), 
        ('execution_time', current_time)])
    graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)])
    graph.executed_operations += 1
    old_end = graph.operation(operation_id, 'end_time')
    new_end = max(current_time, old_end)
    shift = max(0, new_end-old_end)
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', new_end)])
    e = instance.get_item_of_operation(p,o)
    item_id = graph.items_i2g[p][e]
    graph, max_next_operation_end = shift_next_operations(graph, instance, p, e, o, shift)
    graph.update_item(item_id, [
        ('end_time', max(current_time, max_next_operation_end, graph.item(item_id, 'end_time'))),
        ('start_time', min(current_time, graph.item(item_id, 'start_time')))])
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [('end_time', max(current_time, graph.item(ancestor_id, 'end_time')))])
    if instance.is_design[p][o]:
        graph,_ = shift_children_and_operations(graph, instance, p, e, shift)
    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, max_next_operation_end)
    required_types_of_materials[p][o].remove(rt)
    return graph, required_types_of_materials, max(max_ancestors_end, new_end, max_next_operation_end)

def schedule_operation(graph: GraphInstance, instance: Instance, operation_id: int, resource_id: int, required_types_of_resources: list[list[list[int]]], utilization: list[float], current_time: int):
    current_processing_time = graph.need_for_resource(operation_id, resource_id, 'current_processing_time')
    old_end = graph.operation(operation_id, 'end_time')
    operation_end = current_time + current_processing_time
    new_end = max(operation_end, old_end)
    shift = max(0, new_end-old_end)
    p, o = graph.operations_g2i[operation_id]
    e = instance.get_item_of_operation(p, o)
    r = graph.resources_g2i[resource_id]
    rt = instance.get_resource_familly(r)
    estimated_processing_time = instance.operation_resource_time(p, o, rt, max_load=True)
    item_id = graph.items_i2g[p][e]
    graph.inc_resource(resource_id, [('executed_operations', 1), ('remaining_operations', -1)])
    graph.update_resource(resource_id, [('available_time', operation_end)])
    graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', current_time),
        ('end_time', operation_end)])
    utilization[resource_id] += current_processing_time
    graph.executed_operations += 1
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
    graph.update_operation(operation_id, [('end_time', new_end)])
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
        graph.update_item(ancestor_id, [('end_time', operation_end)], maxx=True)
    graph, max_next_operation_end = shift_next_operations(graph, instance, p, e, o, shift)
    graph.update_item(item_id, [('start_time', current_time)], minn=True)
    graph.update_item(item_id, [('end_time', max(operation_end, max_next_operation_end))], maxx=True)
    if not instance.is_design[p][o]:
        graph.inc_item(item_id, [('remaining_physical_time', -estimated_processing_time)])
        if graph.item(item_id, 'remaining_design_time')<= 0:
            graph.executed_items += 1
        for child in instance.get_children(p, e, direct=False):
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    else:
        graph.inc_item(item_id, [('remaining_design_time', -estimated_processing_time)])
        graph,_ = shift_children_and_operations(graph, instance, p, e, shift)
    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, max_next_operation_end)
    return graph, utilization, required_types_of_resources, operation_end, max(operation_end, max_next_operation_end, max_ancestors_end)

def schedule_other_resources_if_simultaneous(instance: Instance, graph: GraphInstance, required_types_of_resources: int, required_types_of_materials: int, utilization: list, operation_id: int, resource_id: int, p: int, o: int, t: int, max_ancestors_end: int, operation_end: int):
    """
        Also schedule other resources if the operation is simultaneous
    """
    not_RT: int = instance.get_resource_familly(graph.resources_g2i[resource_id])
    for rt in instance.required_rt(p, o):
        if rt != not_RT:
            found: bool = False
            for r in instance.resources_by_type(rt):
                if not found:
                    if instance.finite_capacity[r]:
                        other_resource_id = graph.resources_i2g[r]
                        if graph.resource(other_resource_id, 'available_time') <= t:
                            graph, utilization, required_types_of_resources, op_end, _other_max_ancestors_end = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, utilization, t)
                            operation_end = max(operation_end, op_end)
                            max_ancestors_end = max(max_ancestors_end, _other_max_ancestors_end)
                            found = True
                            break
                    else:
                        graph, required_types_of_materials, _other_max_ancestors_end = apply_use_material(graph, instance, operation_id, graph.materials_i2g[r], required_types_of_materials, t)
                        max_ancestors_end = max(max_ancestors_end, _other_max_ancestors_end)
                        found = True
                        break
    return graph, utilization, required_types_of_resources, required_types_of_materials, operation_end, max_ancestors_end

# =====================================================
# =*= III. EXECUTE ONE INSTANCE =*=
# =====================================================

def try_to_open_next_operations(graph: GraphInstance, instance: Instance, previous_operations: list[list[list[int]]], next_operations: list[list[list[int]]], operation_id: int, available_time: int): 
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
                DEBUG_PRINT(f'Enabling operation ({p},{next}) at time {available_time} -> {next_time} in its own timescale...')
                graph.update_operation(next_id, [('is_possible', YES)])
                graph.update_operation(next_id, [('available_time', next_time)], maxx=True)
        if instance.is_last_design(p, e, o):
            for child in instance.get_children(p, e, direct=True):
                child_id = graph.items_i2g[p][child]
                if instance.external[p][child]:
                    DEBUG_PRINT(f'Enabling item {child_id} -> ({p},{child}) for outsourcing (decision yet to make)...')
                graph.update_item(child_id, [('is_possible', YES)])
                graph.update_item(child_id, [('start_time', available_time)], maxx=True)
    return graph

def objective_value(cmax: int, cost: int, cmax_weight: float):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def build_required_resources(i: Instance):
    """
        Build fixed array of required resources per operation
    """
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
    return required_types_of_resources, required_types_of_materials, res_by_types

def policy(probabilities: Tensor, greedy: bool=True):
    return torch.argmax(probabilities.view(-1)).item() if greedy else torch.multinomial(probabilities.view(-1), 1).item()

def update_processing_time(instance: Instance, graph: GraphInstance, op_id: int, res_id: int):
    p, o = graph.operations_g2i[op_id]
    r = graph.resources_g2i[res_id]
    op_setup_time = 0 if (instance.get_operation_type(p, o) == graph.current_operation_type[res_id] or graph.current_operation_type[res_id]<0) else instance.operation_setup[r]
    for d in range(instance.nb_settings):
        op_setup_time += 0 if (graph.current_design_value[res_id][d] == instance.design_value[p][o][d] or graph.current_design_value[res_id][d]<0) else instance.design_setup[r][d] 
    return graph.need_for_resource(op_id, res_id, 'basic_processing_time') + op_setup_time

def next_possible_time(instance: Instance, current_time: int, p: int, o: int):
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if current_time % scale == 0:
        return current_time
    else:
        return ((current_time // scale) + 1) * scale

def manage_queue_of_possible_actions(instance: Instance, graph: GraphInstance, utilization: list, t: int, debug_mode: bool = False):
    """
        Manage the queue of possible actions
    """
    next_date = -1
    operations_to_test = []
    for material_id in graph.loop_materials():
        arrival_time = graph.material(material_id, 'arrival_time')
        if arrival_time>t and (arrival_time<next_date or next_date<0):
            DEBUG_PRINT(f"\t --> New quantity of material {material_id} is available at time {arrival_time} [t={t}]")
            next_date = arrival_time
    for resource_id in graph.loop_resources():
        available_time = graph.resource(resource_id, 'available_time')
        DEBUG_PRINT(f"\t --> Machine {resource_id} is available at time {available_time} [t={t}]")
        if available_time>t and (available_time<next_date or next_date<0):
            next_date = available_time
    for operation_id in graph.loop_operations():
        if graph.operation(operation_id, 'is_possible') == YES and (graph.operation(operation_id, 'remaining_resources')>0 or graph.operation(operation_id, 'remaining_materials')>0):
            available_time = graph.operation(operation_id, 'available_time')
            if available_time <= t:
                p, o = graph.operations_g2i[operation_id]
                available_time = next_possible_time(instance, t, p, o)
            if graph.operation(operation_id, 'remaining_resources')>0:
                DEBUG_PRINT(f"\t --> operation {operation_id} can be scheduled at time {available_time} [t={t}]")
            elif graph.operation(operation_id, 'remaining_materials')>0:
                DEBUG_PRINT(f"\t --> operation {operation_id} can use material at time {available_time} [t={t}]")
            if available_time>t: 
                operations_to_test.append(operation_id)
                if available_time<next_date or next_date<0:
                    next_date = available_time
    if next_date>=0:
        t = next_date
        if t > 0:
            for res_id in graph.loop_resources():
                graph.update_resource(res_id, [('utilization_ratio', utilization[res_id] / t)])
        DEBUG_PRINT(f"New current time t={t}...")
        return graph, utilization, t, False
    else:
        if debug_mode:
            DEBUG_PRINT("End of solving stage!")
            check_completeness(graph, DEBUG_PRINT)
        return graph, utilization, t, True

def solve_one(instance: Instance, agents: list[(Module, str)], train: bool, device: str, debug_mode: bool):
    graph, current_cmax, current_cost, previous_operations, next_operations, related_items, parent_items = translate(i=instance, device=device)
    utilization: list = [0 for _ in graph.loop_resources()]
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    t: int = 0
    alpha: Tensor = torch.tensor([instance.w_makespan], device=device)
    if train:
        training_results: MultiAgent_OneInstance = MultiAgent_OneInstance(
            agent_names=[name for _,name in agents], 
            instance_id=instance.id,
            related_items=related_items,
            parent_items=parent_items,
            w_makespan=alpha,
            device=device)
    first_cmax = -1 * current_cmax * instance.w_makespan
    old_cmax = current_cmax
    old_cost = 0
    terminate = False
    operations_to_test = []
    while not terminate:
        poss_actions, actions_type = get_feasible_actions(instance, graph, operations_to_test, required_types_of_resources, required_types_of_materials, res_by_types, t)
        DEBUG_PRINT(f"Current possible {ACTIONS_NAMES[actions_type]} actions: {poss_actions}")
        if poss_actions:
            if actions_type == SCHEDULING:
                for op_id, res_id in poss_actions:
                    graph.update_need_for_resource(op_id, res_id, [('current_processing_time', update_processing_time(instance, graph, op_id, res_id))])
            if train:
                probs, state_value = agents[actions_type][AGENT](graph.to_state(device=device), poss_actions, related_items, parent_items, alpha)
                idx = policy(probs, greedy=False)
                if actions_type != MATERIAL_USE or graph.material(poss_actions[idx][1], 'remaining_init_quantity')>0:
                    need_reward = True
                    training_results.add_step(
                        agent_name=ACTIONS_NAMES[actions_type], 
                        state=graph.to_state(device=device),
                        probabilities=probs.detach(),
                        actions=poss_actions,
                        id=idx,
                        value=state_value.detach())
                else:
                    need_reward = False
            else:
                with torch.no_grad():
                    probs, state_value = agents[actions_type][AGENT](graph.to_state(device=device), poss_actions, related_items, parent_items, alpha)
                idx = policy(probs, greedy=True)
            if actions_type == OUTSOURCING: # Outsourcing action
                item_id, outsourcing_choice = poss_actions[idx]
                p, e = graph.items_g2i[item_id]
                if outsourcing_choice == YES:
                    graph, _end_date, _local_price = outsource_item(graph, instance, item_id, t, enforce_time=False)
                    graph, current_cmax, current_cost = apply_outsourcing_to_direct_parent(instance, graph, current_cmax, current_cost, previous_operations, item_id, p, e, _end_date, _local_price)
                else:
                    graph, _shifted_project_estimated_end = item_local_production(graph, instance, item_id, p, e)
                    current_cmax = max(current_cmax, _shifted_project_estimated_end)
                if train:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[OUTSOURCING], reward=reward(old_cmax, current_cmax, old_cost, current_cost, a=instance.w_makespan, use_cost=True))
            elif actions_type == SCHEDULING: # scheduling action
                operation_id, resource_id = poss_actions[idx]    
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {t}...")     
                graph, utilization, required_types_of_resources, _operation_end, _max_ancestors_end = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, utilization, t)
                if instance.simultaneous[p][o]:
                    DEBUG_PRINT("\t >> Simulatenous...")
                    graph, utilization, required_types_of_resources, required_types_of_materials, _operation_end, _max_ancestors_end = schedule_other_resources_if_simultaneous(instance, graph, required_types_of_resources, required_types_of_materials, utilization, operation_id, resource_id, p, o, t, _max_ancestors_end, _operation_end)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, _operation_end)
                current_cmax = max(current_cmax, _max_ancestors_end)
                DEBUG_PRINT(f"End of scheduling at time {_operation_end} [NEW CMAX = {current_cmax} - COST = {current_cost}]...")
                if train:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[SCHEDULING], reward=reward(old_cmax, current_cmax, a=instance.w_makespan))
            else: # Material use action
                operation_id, material_id = poss_actions[idx]
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]}...")  
                graph, required_types_of_materials, _max_ancestors_end = apply_use_material(graph, instance, operation_id, material_id, required_types_of_materials, t)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, t)
                current_cmax = max(current_cmax, _max_ancestors_end)
                if train and need_reward:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[MATERIAL_USE], reward=reward(old_cmax, current_cmax, a=instance.w_makespan))
            old_cost = current_cost
            old_cmax = current_cmax
        else: # No more possible action at time t
            graph, utilization, t, terminate = manage_queue_of_possible_actions(instance, graph, utilization, t, debug_mode)
    if train:
        training_results.update_last_reward(agent_name=ACTIONS_NAMES[SCHEDULING], init_cmax=first_cmax)
        return training_results, graph, current_cmax, current_cost
    else:
        return graph, current_cmax, current_cost

# =====================================================
# =*= IV. MAIN CODE =*=
# =====================================================

def load_trained_models(model_path:str, run_number:int, device:str, fine_tuned: bool = False, size: str = "", id: str = ""):
    index = str(run_number)
    base_name = f"{size}_{id}_" if fine_tuned else ""
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['embedding_hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(model_path+'/'+base_name+'gnn_weights_'+index+'.pth', map_location=torch.device(device)))
    shared_critic: L1_CommonCritic = L1_CommonCritic(GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['value_hidden_channels'])
    shared_critic.load_state_dict(torch.load(model_path+'/'+base_name+'critic_weights_'+index+'.pth', map_location=torch.device(device)))
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    scheduling_actor: L1_SchedulingActor = L1_SchedulingActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    outsourcing_actor.load_state_dict(torch.load(model_path+'/'+base_name+'outsourcing_weights_'+index+'.pth', map_location=torch.device(device)))
    scheduling_actor.load_state_dict(torch.load(model_path+'/'+base_name+'scheduling_weights_'+index+'.pth', map_location=torch.device(device)))
    material_actor.load_state_dict(torch.load(model_path+'/'+base_name+'material_use_weights_'+index+'.pth', map_location=torch.device(device)))
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

def init_new_models():
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['embedding_hidden_channels'], GNN_CONF['nb_layers'])
    shared_critic: L1_CommonCritic = L1_CommonCritic(GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['value_hidden_channels'])
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    scheduling_actor: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['actor_hidden_channels'])
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

def pre_train_on_all_instances(run_number: int, device: str, path: str, debug_mode: bool):
    """
        Pre-train networks on all instances
    """
    first = (_run_number<=1)
    previous_run = run_number - 1
    agents, shared_embbeding_stack, shared_critic = init_new_models() if first else load_trained_models(model_path=path+directory.models, run_number=previous_run, device=device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for agent,_ in agents:
        agent = agent.to(device)
    optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(shared_embbeding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[SCHEDULING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), 
        lr=LEARNING_RATE)
    if not first:
        optimizer.load_state_dict(torch.load(path+directory.models+"/adam_"+str(previous_run)+".pth"))
    print("Pre-training models with MAPPO (on several instances)...")
    PPO_pre_train(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, optimizer=optimizer, path=path, solve_function=solve_one, device=device, run_number=run_number, debug_mode=debug_mode)

def fine_tune_on_target(id: str, size: str, pre_trained_number: int, path: str, debug_mode: bool, device: str, use_pre_train: bool = False, interactive: bool = True):
    """
        Fine-tune on target instance (size, id)
    """
    agents, shared_embbeding_stack, shared_critic = init_new_models() if not use_pre_train else load_trained_models(model_path=path+directory.models, run_number=pre_trained_number, device=device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for agent,_ in agents:
        agent = agent.to(device)
    optimizer = torch.optim.Adam(
        list(shared_critic.parameters()) + list(shared_embbeding_stack.parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[SCHEDULING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), 
        lr=LEARNING_RATE)
    print("Fine-tuning models with MAPPO (on target instance)...")
    PPO_fine_tuning(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, optimizer=optimizer, path=path, solve_function=solve_one, device=device, id=id, size=size, interactive=interactive, debug_mode=debug_mode)

def solve_only_target(id: str, size: str, run_number: int, device: str, debug_mode: bool, path: str):
    """
        Solve the target instance (size, id) only using inference
    """
    print(f"SOLVE TARGET INSTANCE {size}_{id}...")
    target_instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    DEBUG_PRINT(target_instance.display())
    start_time = systime.time()
    first = (_run_number<=1)
    agents, shared_embbeding_stack, shared_critic = init_new_models() if first else load_trained_models(model_path=path+directory.models, run_number=run_number, device=device)
    for agent,_ in agents:
        agent = agent.to(device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    graph, current_cmax, current_cost = solve_one(target_instance, agents, train=False, device=device, debug_mode=debug_mode)
    final_metrics = pd.DataFrame({
        'index': [target_instance.id],
        'value': [objective_value(current_cmax, current_cost, target_instance.w_makespan)/100], 
        'computing_time': [systime.time()-start_time],
        'device_used': [device]
    })
    solution: HeuristicSolution = translate_solution(graph, target_instance)
    print(final_metrics)
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_gns_'+id+'.csv', index=False)
    with open(directory.solutions+'/'+size+'/gns_'+str(run_number)+'_graph_'+id+'.pkl', 'wb') as f:
            pickle.dump(graph, f)
    with open(directory.solutions+'/'+size+'/gns_'+str(run_number)+'_solution_'+id+'.pkl', 'wb') as f:
            pickle.dump(solution, f)

def solve_all_instances(run_number: int, device: str, debug_mode: bool, path: str):
    """
        Solve all instances only in inference mode
    """
    instances: list[Instance] = load_training_dataset(path=path, train=False, debug_mode=debug_mode)
    for i in instances:
        if (i.size, i.id) not in [('s', 172)]:
            solve_only_target(id=str(i.id), size=str(i.size), run_number=run_number, device=device, debug_mode=debug_mode, path=path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 GNS solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--target", help="Do you want to load a pre-trained model", required=False)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    parser.add_argument("--path", help="Saving path on the server", required=True)
    parser.add_argument("--use_pretrain", help="Use a pre-train model while fine-tuning", required=False)
    parser.add_argument("--interactive", help="Display losses, cmax, and cost in real-time or not", required=False)
    parser.add_argument("--number", help="The number of the current run", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    _debug_mode = (args.mode == 'test')
    _run_number = int(args.number)
    _device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TPU Device: {_device}...")
    DEBUG_PRINT = debug_printer(_debug_mode)
    if to_bool(args.train):
        if to_bool(args.target):
            # USE OF PRE-TRAIN MODEL: python gns_solver.py --train=true --target=true --size=s --id=151 --mode=prod --use_pretrain=true --interactive=false --number=1 --path=./ 
            # NEW MODEL: python gns_solver.py --train=true --target=true --size=s --id=151 --mode=prod --use_pretrain=false --interactive=true --number=1 --path=./
            # TRY ON DEBUG INSTANCE: python gns_solver.py --train=true --target=true --size=d --id=debug --mode=prod --use_pretrain=false --interactive=true --number=1 --path=./
            fine_tune_on_target(id=args.id, size=args.size, pre_trained_number=_run_number, path=args.path, debug_mode=_debug_mode, device=_device, use_pre_train=to_bool(args.use_pretrain), interactive=to_bool(args.interactive))
        else:
            # python gns_solver.py --train=true --target=false --mode=test --number=1 --path=./
            pre_train_on_all_instances(run_number=_run_number, path=args.path, debug_mode=_debug_mode, device=_device)
    else:
        if to_bool(args.target):
            # SOLVE ACTUAL INSTANCE: python gns_solver.py --target=true --size=s --id=151 --train=false --mode=test --path=./ --number=1
            # TRY ON DEBUG INSTANCE: python gns_solver.py --target=true --size=d --id=debug --train=false --mode=test --path=./ --number=1
            solve_only_target(id=args.id, size=args.size, run_number=args.number, device=_device, debug_mode=_debug_mode, path=args.path)
        else:
            # python gns_solver.py --train=false --target=false --mode=test --path=./ --number=1
            solve_all_instances(run_number=args.number, device=_device, debug_mode=_debug_mode, path=args.path)
    print("===* END OF FILE *===")