import argparse
from model.instance import Instance
from model.graph import GraphInstance, NO, NOT_YET, YES
from model.gnn import L1_EmbbedingGNN, L1_MaterialActor, L1_OutousrcingActor, L1_SchedulingActor, L1_CommonCritic
from common import load_instance, to_bool, directory
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
import time as systime
from typing import Callable
from torch import Tensor
from torch.nn import Module
from instance2graph_translator import translate
from debug.debug_gns import check_completeness, debug_printer
from gns_ppo_trainer import reward, PPO_train
from model.agent import MultiAgent_OneInstance

# =====================================================
# =*= 1st MAIN FILE OF THE PROJECT: GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
GNN_CONF = {
    'embedding_size': 16,
    'nb_layers': 2,
    'hidden_channels': 128
}
AC_CONF = {
    'hidden_channels': 64
}

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

def get_scheduling_and_material_use_actions(instance: Instance, graph: GraphInstance, operations: list[int], required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], res_by_types: list[list[int]], current_time: int, debug_print: Callable):
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

def get_feasible_actions(instance: Instance, graph: GraphInstance, operations: list[int], required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], res_by_types: list[list[int]], current_time: int, debug_print: Callable):
    actions = get_outourcing_actions(instance, graph)
    type = OUTSOURCING
    if not actions:
        scheduling_actions, material_use_actions = get_scheduling_and_material_use_actions(instance, graph, operations, required_types_of_resources, required_types_of_materials, res_by_types, current_time, debug_print)
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
    for next in range(o, end):
        graph, end_next = shift_one_operation(graph, instance, p, next, shift)
        max_end = max(end_next, max_end)
    return graph, max_end

def shift_children_and_operations(graph: GraphInstance, instance: Instance, p: int, e: int, shift: int):
    if shift <= 0:
        return graph, 0
    max_child_end = 0
    for child in instance.get_children(p, e, direct=False):
        child_id = graph.items_i2g[p][child]
        graph.inc_item(child_id, [
            ('start_time', shift),
            ('end_time', shift)
        ])
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
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('end_time', shift)])
        max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
        for o in instance.loop_item_operations(p, ancestor):
            if not instance.is_design[p][o]:
                graph,_ = shift_one_operation(graph, instance, p, o, shift)
                max_ancestor_end = max(max_ancestor_end, graph.item(ancestor_id, 'end_time'))
    return graph, max_ancestor_end

def item_local_production(graph: GraphInstance, instance: Instance, item_id: int, p: int, e: int, debug_print: Callable):
    debug_print(f"Producing item {item_id} -> ({p},{e}) locally...")
    old_end = graph.item(item_id, 'end_time')
    max_child_end = 0
    design_mean_time, physical_mean_time = instance.item_processing_time(p, e, total_load=False)
    graph, max_child_end = shift_children_and_operations(graph, instance, p, e, design_mean_time)
    new_end = physical_mean_time + (max_child_end if max_child_end>0 else (graph.item(item_id, 'start_time')+design_mean_time))
    graph, max_ancestors_end = shift_ancestors_and_operations(graph, instance, p, e, shift=new_end-old_end)
    graph.update_item(item_id, [('outsourced', NO), ('end_time', new_end)])
    return graph, max(new_end, max_ancestors_end)

def outsource_item(graph: GraphInstance, instance: Instance, item_id: int, t: int, enforce_time: bool=False):
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
    for child in instance.get_children(p, e, direct=True):
        graph, child_time, child_cost = outsource_item(graph, instance, graph.items_i2g[p][child], outsourcing_start_time, enforce_time=True)
        cost += child_cost
        end_date = max(t, child_time)
    return graph, end_date, cost

def apply_use_material(graph: GraphInstance, instance: Instance, operation_id: int, material_id: int, required_types_of_materials:list[list[list[int]]], current_time: int):
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
    old_end = graph.operation(operation_id, 'end_time')
    new_end = max(current_time, old_end)
    shift = max(0, new_end-old_end)
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', new_end)
    ])
    e = instance.get_item_of_operation(p,o)
    item_id = graph.items_i2g[p][e]
    graph, max_next_operation_end = shift_next_operations(graph, instance, p, e, o, shift)
    graph.update_item(item_id, [
        ('end_time', max(current_time, max_next_operation_end, graph.item(item_id, 'end_time'))),
        ('start_time', min(current_time, graph.item(item_id, 'start_time')))
    ])
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [('end_time', max(current_time, graph.item(ancestor_id, 'end_time')))])
    if instance.is_design[p][o]:
        graph,_ = shift_children_and_operations(graph, instance, p, e, shift)
    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, max_next_operation_end)
    required_types_of_materials[p][o].remove(rt)
    return graph, required_types_of_materials, max(max_ancestors_end, new_end)

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
    graph.update_operation(operation_id, [('end_time', new_end)])
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
        graph.update_item(ancestor_id, [('end_time', max(operation_end, graph.item(ancestor_id, 'end_time')))])
    graph, max_next_operation_end = shift_next_operations(graph, instance, p, e, o, shift)
    graph.update_item(item_id, [
        ('start_time', min(current_time, graph.item(item_id, 'start_time'))),
        ('end_time', max(operation_end, max_next_operation_end, graph.item(item_id, 'end_time')))
    ])
    if not instance.is_design[p][o]:
        graph.inc_item(item_id, [('remaining_physical_time', -estimated_processing_time)])
        for child in instance.get_children(p, e, direct=False):
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    else:
        graph.inc_item(item_id, [('remaining_design_time', -estimated_processing_time)])
        graph,_ = shift_children_and_operations(graph, instance, p, e, shift)
    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, max_next_operation_end)
    return graph, utilization, required_types_of_resources, operation_end, max(operation_end, max_ancestors_end)

# =====================================================
# =*= III. EXECUTE ONE INSTANCE =*=
# =====================================================

def try_to_open_next_operations(graph: GraphInstance, instance: Instance, previous_operations: list[list[list[int]]], next_operations: list[list[list[int]]], operation_id: int, available_time: int, debug_print: Callable): 
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

def objective_value(cmax: int, cost: int, cmax_weight: float):
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

def solve_one(instance: Instance, agents: list[(Module, str)], path: str="", train: bool=False, device:str = 'cpu', debug_mode: bool=False):
    debug_print = debug_printer(debug_mode)
    debug_print(instance.display())
    start_time = systime.time()
    graph, current_cmax, current_cost = translate(instance, device)
    utilization = [0 for _ in graph.loop_resources()]
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    previous_operations, next_operations = instance.build_next_and_previous_operations()
    t = 0
    related_items = graph.flatten_related_items()
    parent_items = graph.flatten_parents()
    alpha = torch.tensor([instance.w_makespan], device=device)
    related_items.to(device)
    parent_items.to(device)
    if train:
        training_results: MultiAgent_OneInstance = MultiAgent_OneInstance(
            agent_names=[name for _,name in agents], 
            instance_id=instance.id,
            related_items=related_items,
            parent_items=parent_items,
            w_makespan=alpha,
            device=device)
    old_cmax = current_cmax
    old_cost = current_cost
    terminate = False
    operations_to_test = []
    while not terminate:
        poss_actions, actions_type = get_feasible_actions(instance, graph, operations_to_test, required_types_of_resources, required_types_of_materials, res_by_types, t, debug_print)
        debug_print(f"Current possible actions: {poss_actions}")
        if poss_actions:
            if actions_type == SCHEDULING:
                for op_id, res_id in poss_actions:
                    graph.update_need_for_resource(op_id, res_id, [('current_processing_time', update_processing_time(instance, graph, op_id, res_id))])
            probs, state_value = agents[actions_type][AGENT](graph.to_state(device=device), poss_actions, related_items, parent_items, alpha)
            idx = policy(probs, greedy=(not train))
            if train:
                training_results.add_step(
                    agent_name=ACTIONS_NAMES[actions_type], 
                    state=graph.to_state(device=device),
                    probabilities=probs.detach(),
                    actions=poss_actions,
                    id=idx,
                    value=state_value.detach())
            if actions_type == OUTSOURCING:
                item_id, outsourcing_choice = poss_actions[idx]
                p, e = graph.items_g2i[item_id]
                if outsourcing_choice == YES:
                    debug_print(f"Outsourcing item {item_id} -> ({p},{e})...")
                    graph, end_date, local_price = outsource_item(graph, instance, item_id, t, enforce_time=False)
                    approximate_design_load, approximate_physical_load = instance.item_processing_time(p, e, total_load=True)
                    for ancestor in instance.get_ancestors(p, e):
                        ancestor_id = graph.items_i2g[p][ancestor]
                        graph.update_item(ancestor_id, [
                            ('children_time', graph.item(ancestor_id, 'children_time')-(approximate_design_load+approximate_physical_load)),
                        ])
                    graph, max_ancestors_end = shift_ancestors_physical_operations(graph, instance, p, e, end_date)
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
                    current_cmax = max(current_cmax, max_ancestors_end)
                else:
                    graph, shifted_project_estimated_end = item_local_production(graph, instance, item_id, p, e, debug_print)
                    current_cmax = max(current_cmax, shifted_project_estimated_end)
                if train:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[OUTSOURCING], reward=reward(old_cmax, current_cmax, old_cost, current_cost, a=instance.w_makespan, use_cost=True))
            elif actions_type == SCHEDULING:
                operation_id, resource_id = poss_actions[idx]    
                p, o = graph.operations_g2i[operation_id]
                debug_print(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {t}...")     
                graph, utilization, required_types_of_resources, operation_end, max_ancestors_end = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, utilization, t)
                debug_print(f"End of scheduling at time {operation_end}...")
                if instance.simultaneous[p][o]:
                    for rt in instance.required_rt(p, o):
                        if rt != instance.get_resource_familly(graph.resources_g2i[resource_id]):
                            for r in instance.resources_by_type(rt):
                                if instance.finite_capacity[r]:
                                    other_resource_id = graph.resources_i2g[r]
                                    if graph.resource(other_resource_id, 'available_time') <= t:
                                        graph, utilization, required_types_of_resources, op_end, other_max_ancestors_end = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, utilization, t)
                                        operation_end = max(operation_end, op_end)
                                        max_ancestors_end = max(max_ancestors_end, other_max_ancestors_end)
                                        break
                                else:
                                    graph, required_types_of_materials, other_max_ancestors_end = apply_use_material(graph, instance, operation_id, graph.materials_i2g[r], required_types_of_materials, t)
                                    max_ancestors_end = max(max_ancestors_end, other_max_ancestors_end)
                                    break
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, operation_end, debug_print)
                current_cmax = max(current_cmax, max_ancestors_end)
                if train:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[SCHEDULING], reward=reward(old_cmax, current_cmax))
            else:
                operation_id, material_id = poss_actions[idx]
                p, o = graph.operations_g2i[operation_id]
                debug_print(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]}...")  
                graph, required_types_of_materials, max_ancestors_end = apply_use_material(graph, instance, operation_id, material_id, required_types_of_materials, t)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, t, debug_print)
                current_cmax = max(current_cmax, max_ancestors_end)
                if train:
                    training_results.add_reward(agent_name=ACTIONS_NAMES[MATERIAL_USE], reward=reward(old_cmax, current_cmax))
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
                if debug_mode:
                    debug_print("End of solving stage!")
                    check_completeness(graph, debug_print)
                terminate = True
    if train:
        return training_results
    else:
        solutions_df = pd.DataFrame({
            'index': [instance.id],
            'value': [objective_value(current_cmax, current_cost, instance.w_makespan)/100], 
            'computing_time': [systime.time()-start_time]
        })
        print(solutions_df)
        solutions_df.to_csv(path, index=False)
        return current_cmax, current_cost 

# =====================================================
# =*= IV. MAIN CODE =*=
# =====================================================

def load_trained_models(model_path):
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(model_path+'/gnn_weights.pth'))
    shared_critic: L1_CommonCritic = L1_CommonCritic(GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    shared_critic.load_state_dict(torch.load(model_path+'/critic_weights.pth'))
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    scheduling_actor: L1_SchedulingActor = L1_SchedulingActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    outsourcing_actor.load_state_dict(torch.load(model_path+'/outsourcing_weights.pth'))
    scheduling_actor.load_state_dict(torch.load(model_path+'/scheduling_weights.pth'))
    material_actor.load_state_dict(torch.load(model_path+'/material_weights.pth'))
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

def init_new_models():
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_critic: L1_CommonCritic = L1_CommonCritic(GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    scheduling_actor: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, GNN_CONF['embedding_size'], AC_CONF['hidden_channels'])
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    parser.add_argument("--path", help="Saving path on the server", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    debug_mode = (args.mode == 'test')
    if to_bool(args.train):
        '''
            Test training mode with: bash _env.sh
            python gns_solver.py --train=true --mode=test --path=./
        '''
        agent, shared_embbeding_stack, shared_critic = init_new_models()
        PPO_train(agent, shared_embbeding_stack, shared_critic, path=args.path, solve_function=solve_one, debug_mode=debug_mode)
    else:
        '''
            Test inference mode with: bash _env.sh
            python gns_solver.py --size=s --id=151 --train=false --mode=test --path=./
        '''
        print(f"SOLVE TARGET INSTANCE {args.size}_{args.id}...")
        instance: Instance = load_instance(args.path+directory.instances+'/test/'+args.size+'/instance_'+args.id+'.pkl')
        agents, shared_embbeding_stack, shared_critic = init_new_models() if args.mode == 'test' else load_trained_models(args.path+directory.models) 
        device = "cuda" if not debug_mode and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            for agent,_ in agents:
                agent.to(device)
            shared_embbeding_stack.to(device)
            shared_critic.to(device)
        solve_one(instance, agents, path=args.path+directory.instances+'/test/'+args.size+'/solution_gns_'+args.id+'.csv', train=False, device=device, debug_mode=debug_mode)
    print("===* END OF FILE *===")