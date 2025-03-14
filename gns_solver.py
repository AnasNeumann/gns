import argparse
from model.instance import Instance
from model.graph import GraphInstance, State, NO, NOT_YET, YES
from model.gnn import L1_EmbbedingGNN, L1_MaterialActor, L1_OutousrcingActor, L1_SchedulingActor, L1_CommonCritic
from model.solution import HeuristicSolution
from tools.common import load_instance, to_bool, directory
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
import time as systime
from torch import Tensor
from torch.nn import Module
from translators.instance2graph_translator import translate
from translators.graph2solution_translator import translate_solution
from debug.debug_gns import debug_printer
from multi_stage_pre_training import multi_stage_pre_train, load_training_dataset
from model.agent import MultiAgent_OneInstance
import pickle
from multi_stage_ppo_tuning import multi_stage_fine_tuning
from model.reward_memory import Memory, Decision

# =====================================================
# =*= 1st MAIN FILE OF THE PROJECT: GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

DEBUG_PRINT: callable = None
LEARNING_RATE = 1e-4
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
SOLVING_REPETITIONS = 10
GNN_CONF = {
    'resource_and_material_embedding_size': 8,
    'operation_and_item_embedding_size': 12,
    'nb_layers': 2,
    'embedding_hidden_channels': 64,
    'value_hidden_channels': 128,
    'actor_hidden_channels': 128}

# =====================================================
# =*= I. SEARCH FOR FEASIBLE ACTIONS =*=
# =====================================================

def reccursive_outourcing_actions(instance: Instance, graph: GraphInstance, item_id: int):
    """
        Search possible outsourcing actions (sub-loops)
    """
    actions = []
    p, e = graph.items_g2i[item_id]
    external: bool = instance.external[p][e]
    decision_made = graph.item(item_id, 'outsourced')
    available = graph.item(item_id, 'is_possible')
    if available==YES:
        if external and decision_made==NOT_YET:
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
        elif not external or decision_made==NO:
            for child in graph.get_direct_children(instance, item_id):
                actions.extend(reccursive_outourcing_actions(instance, graph, child))
    return actions

def get_outourcing_actions(instance: Instance, graph: GraphInstance):
    """
        Search possible outsourcing actions
    """
    actions = []
    for project_head in graph.project_heads:
        actions.extend(reccursive_outourcing_actions(instance, graph, project_head))
    return actions

def get_scheduling_and_material_use_actions(instance: Instance, graph: GraphInstance, operations: list[int], required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], res_by_types: list[list[int]], current_time: int):
    """
        Search possible material use and scheduling actions
    """
    scheduling_actions = []
    material_use_actions = []
    ops_to_test = operations if operations else graph.loop_operations()
    for operation_id in ops_to_test:
        p, o = graph.operations_g2i[operation_id]
        e = instance.get_item_of_operation(p, o)
        item_id = graph.items_i2g[p][e]
        timescale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
        if graph.item(item_id, 'is_possible')==YES \
                and graph.item(item_id, 'outsourced')==NO \
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
                                sync_available = False
                                break
                    if not sync_available:
                        break
                if instance.simultaneous[p][o] and sync_available:
                    for rt in required_types_of_materials[p][o]:
                        for m in res_by_types[rt]:
                            mat_id = graph.materials_i2g[m]
                            if instance.purchase_time[m] > current_time and graph.material(mat_id, 'remaining_init_quantity') < instance.quantity_needed[m][p][o]:
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
    """
        Search next possible actions with priority between decision spaces (outsourcing >> scheduling >> material use)
    """
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
        ('remaining_time', 0),
        ('children_time', 0),
        ('start_time', outsourcing_start_time),
        ('end_time', end_date)])
    for o in instance.loop_item_operations(p,e):
        op_id = graph.operations_i2g[p][o]
        graph.update_operation(op_id, [
            ('remaining_resources', 0),
            ('remaining_materials', 0),
            ('remaining_time', 0)])
        for rt in instance.required_rt(p, o):
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
        end_date = max(end_date, child_time)
    return graph, end_date, cost

def apply_outsourcing_to_direct_parent(instance: Instance, graph: GraphInstance, previous_operations: list, p: int, e: int, end_date: int):
    """
        Apply an outsourcing decision to the direct parent
    """
    approximate_design_load, approximate_physical_load = instance.item_processing_time(p, e, total_load=True)
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [
            ('children_time', graph.item(ancestor_id, 'children_time')-(approximate_design_load+approximate_physical_load))])
    _parent = instance.get_direct_parent(p, e)
    for o in instance.first_physical_operations(p, _parent):
        next_good_to_go: bool = True
        _t = next_possible_time(instance, end_date, p, o)
        graph.update_operation(graph.operations_i2g[p][o], [('available_time', _t)], maxx=True)
        for previous in previous_operations[p][o]:
            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                DEBUG_PRINT(f"\t >> Cannot open parent' first physical operation ({p},{o}) due to ({p},{previous}) not finished! BUT IT WILL BE MOVED TO AT LEAST {_t}")
                next_good_to_go = False
                break
        if next_good_to_go:
            DEBUG_PRINT(f"\t >> Opening first physical operation ({p},{o}) of parent {_parent} at {_t}!")
            graph.update_operation(graph.operations_i2g[p][o], [('is_possible', True)])
    return graph

def apply_use_material(graph: GraphInstance, instance: Instance, operation_id: int, material_id: int, required_types_of_materials:list[list[list[int]]], current_time: int):
    """
        Apply use material to an operation
    """
    p, o = graph.operations_g2i[operation_id]
    rt = instance.get_resource_familly(graph.materials_g2i[material_id])
    quantity_needed = graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity = graph.material(material_id, 'remaining_init_quantity')
    waiting_demand = graph.material(material_id, 'remaining_demand') 
    graph.update_need_for_material(operation_id, material_id, [('status', YES), ('execution_time', current_time)])
    graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)])
    old_end = graph.operation(operation_id, 'end_time')
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', max(current_time, old_end))])
    required_types_of_materials[p][o].remove(rt)
    return graph, required_types_of_materials

def schedule_operation(graph: GraphInstance, instance: Instance, operation_id: int, resource_id: int, required_types_of_resources: list[list[list[int]]], utilization: list[float], current_time: int):
    """
        Schedule an operation on a resource
    """
    current_processing_time = graph.need_for_resource(operation_id, resource_id, 'current_processing_time')
    operation_end = current_time + current_processing_time
    p, o = graph.operations_g2i[operation_id]
    e = instance.get_item_of_operation(p, o)
    r = graph.resources_g2i[resource_id]
    rt = instance.get_resource_familly(r)
    estimated_processing_time = instance.operation_resource_time(p, o, rt, max_load=True)
    item_id = graph.items_i2g[p][e]
    graph.inc_resource(resource_id, [('remaining_operations', -1)])
    graph.update_resource(resource_id, [('available_time', operation_end)])
    graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', current_time),
        ('end_time', operation_end)])
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
    graph.update_operation(operation_id, [('end_time', operation_end)], maxx=True)
    graph.update_item(item_id, [('start_time', current_time)], minn=True)
    graph.update_item(item_id, [('end_time', operation_end)], maxx=True)
    graph.inc_item(item_id, [('remaining_time', -estimated_processing_time)])
    for ancestor in instance.get_ancestors(p, e):
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
    if not instance.is_design[p][o]:
        for child in instance.get_children(p, e, direct=False):
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    return graph, utilization, required_types_of_resources, operation_end

def schedule_other_resources_if_simultaneous(instance: Instance, graph: GraphInstance, required_types_of_resources: int, required_types_of_materials: int, utilization: list, operation_id: int, resource_id: int, p: int, o: int, t: int, operation_end: int):
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
                            graph, utilization, required_types_of_resources, op_end = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, utilization, t)
                            operation_end = max(operation_end, op_end)
                            found = True
                            break
                    else:
                        graph, required_types_of_materials = apply_use_material(graph, instance, operation_id, graph.materials_i2g[r], required_types_of_materials, t)
                        found = True
                        break
    return graph, utilization, required_types_of_resources, required_types_of_materials, operation_end

def try_to_open_next_operations(graph: GraphInstance, instance: Instance, previous_operations: list[list[list[int]]], next_operations: list[list[list[int]]], operation_id: int, available_time: int): 
    """
        Try to open next operations after finishing using a resource or material
    """
    if graph.is_operation_complete(operation_id):
        p, o = graph.operations_g2i[operation_id]
        e = instance.get_item_of_operation(p, o)
        for next in next_operations[p][o]:
            next_good_to_go = True
            next_id = graph.operations_i2g[p][next]
            for previous in previous_operations[p][next]:
                if previous != operation_id and not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                    next_good_to_go = False
                    break
            next_time = next_possible_time(instance, available_time, p, next)
            graph.update_operation(next_id, [('available_time', next_time)], maxx=True)
            if next_good_to_go:
                DEBUG_PRINT(f'Enabling operation ({p},{next}) at time {available_time} -> {next_time} in its own timescale...')
                graph.update_operation(next_id, [('is_possible', YES)])
        if instance.is_last_design(p, e, o):
            for child in instance.get_children(p, e, direct=True):
                child_id = graph.items_i2g[p][child]
                if instance.external[p][child]:
                    DEBUG_PRINT(f'Enabling item {child_id} -> ({p},{child}) for outsourcing (decision yet to make)...')
                graph.update_item(child_id, [('is_possible', YES)])
                graph.update_item(child_id, [('start_time', available_time)], maxx=True)
    return graph

# ================================
# =*= III. AUXILIARY FUNCTIONS =*=
# ================================

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

def objective_value(cmax: int, cost: int, cmax_weight: float):
    """
        Compute the final objective value (to compare with other solving methos)
    """
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def policy(probabilities: Tensor, greedy: bool=True):
    """
        Select one action based on current policy
    """
    return torch.argmax(probabilities.view(-1)).item() if greedy else torch.multinomial(probabilities.view(-1), 1).item()

def update_processing_time(instance: Instance, graph: GraphInstance, op_id: int, res_id: int):
    """
        Update processing times with current design settings and operation types of each finite-capacity resources
    """
    p, o = graph.operations_g2i[op_id]
    r = graph.resources_g2i[res_id]
    basic_processing_time = instance.execution_time[r][p][o]
    op_setup_time = 0 if (instance.get_operation_type(p, o) == graph.current_operation_type[res_id] or graph.current_operation_type[res_id]<0) else instance.operation_setup[r]
    for d in range(instance.nb_settings):
        op_setup_time += 0 if (graph.current_design_value[res_id][d] == instance.design_value[p][o][d] or graph.current_design_value[res_id][d]<0) else instance.design_setup[r][d] 
    return basic_processing_time + op_setup_time

def next_possible_time(instance: Instance, current_time: int, p: int, o: int):
    """
        Search the next possible execution time with correct timescale of the operation
    """
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if current_time % scale == 0:
        return current_time
    else:
        return ((current_time // scale) + 1) * scale

def manage_current_time(graph: GraphInstance, instance: Instance, utilization: list, t: int):
    """
        Manage the queue of possible actions
    """
    next_date = -1
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
            if available_time>t:
                DEBUG_PRINT(f"\t --> operation {operation_id} could be scheduled at {available_time} [t={t}]")
                if (available_time<next_date or next_date<0):
                    next_date = available_time
            else:
                p, o = graph.operations_g2i[operation_id]
                _next_possible_date = next_possible_time(instance, t, p, o)
                DEBUG_PRINT(f"\t --> operation {operation_id} can be scheduled since {available_time}, next possible date is {_next_possible_date} [t={t}]")
                if _next_possible_date>t and (_next_possible_date<next_date or next_date<0):
                    next_date = _next_possible_date
    if next_date>=0:
        t = next_date
        if t > 0:
            for res_id in graph.loop_resources():
                graph.update_resource(res_id, [('utilization_ratio', utilization[res_id] / t)])
        DEBUG_PRINT(f"New current time t={t}...")
        return graph, utilization, t, False
    else:
        return graph, utilization, t, True
        
# =====================================================
# =*= IV. EXECUTE ONE INSTANCE =*=
# =====================================================

def solve_one(instance: Instance, agents: list[(Module, str)], trainable: list, train: bool, device: str, greedy: bool = False, fixed_alpha: float = -1, reward_MEMORY: Memory = None):
    graph, lb_cmax, lb_cost, previous_operations, next_operations, related_items, parent_items = translate(i=instance, device=device)
    utilization: list = [0 for _ in graph.loop_resources()]
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    t: int = 0
    _agents_names: list[str] = []
    outsourcing_training_stage: bool = False
    scheduling_training_stage: bool = False
    material_use_training_stage: bool = False
    alpha: Tensor = torch.tensor([instance.w_makespan], device=device) if fixed_alpha < 0 else torch.tensor([fixed_alpha], device=device)
    if train:
        if trainable[OUTSOURCING]:
            _agents_names.append(ACTIONS_NAMES[OUTSOURCING])
            outsourcing_training_stage = True
        if trainable[SCHEDULING]:
            _agents_names.append(ACTIONS_NAMES[SCHEDULING])
            scheduling_training_stage = True
        if trainable[MATERIAL_USE]:
            _agents_names.append(ACTIONS_NAMES[MATERIAL_USE])
            material_use_training_stage = True
        alpha = alpha if outsourcing_training_stage else torch.tensor([1.0], device=device)
        _local_decisions: list[Decision] = []
        training_results: MultiAgent_OneInstance = MultiAgent_OneInstance(
            agent_names=_agents_names, 
            instance_id=instance.id,
            related_items=related_items,
            parent_items=parent_items,
            w_makespan=alpha,
            device=device)
    current_cmax = 0
    current_cost = 0
    old_cmax = 0
    old_cost = 0
    terminate = False
    operations_to_test = []
    DEBUG_PRINT(f"Init Cmax: {lb_cmax} - Init cost: {lb_cost}$")
    while not terminate:
        poss_actions, actions_type = get_feasible_actions(instance, graph, operations_to_test, required_types_of_resources, required_types_of_materials, res_by_types, t)
        DEBUG_PRINT(f"Current possible {ACTIONS_NAMES[actions_type]} actions: {poss_actions}")
        if poss_actions:
            if actions_type == SCHEDULING:
                for op_id, res_id in poss_actions:
                    graph.update_need_for_resource(op_id, res_id, [('current_processing_time', update_processing_time(instance, graph, op_id, res_id))])
            if (actions_type == OUTSOURCING) and not outsourcing_training_stage and len(poss_actions) > 1:
                poss_actions = [ax for ax in poss_actions if ax[1] == NO]
            if (actions_type == MATERIAL_USE) and not material_use_training_stage and len(poss_actions) > 1: 
                poss_actions = poss_actions[:1]
            if train:
                state: State = graph.to_state(device=device)
                probs, state_value = agents[actions_type][AGENT](state, poss_actions, related_items, parent_items, alpha)
                idx = policy(probs, greedy=False)
                if (actions_type==SCHEDULING and scheduling_training_stage) \
                        or (actions_type==OUTSOURCING and outsourcing_training_stage) \
                        or (actions_type== MATERIAL_USE and material_use_training_stage and graph.material(poss_actions[idx][1],'remaining_init_quantity')>0):
                    need_reward = True
                    training_results.add_step(
                        agent_name=ACTIONS_NAMES[actions_type], 
                        state=state,
                        probabilities=probs.detach(),
                        actions=poss_actions,
                        id=idx,
                        value=state_value.detach())
                else:
                    need_reward = False
            else:
                with torch.no_grad():
                    probs, state_value = agents[actions_type][AGENT](graph.to_state(device=device), poss_actions, related_items, parent_items, alpha)
                idx = policy(probs, greedy=greedy)
            if actions_type == OUTSOURCING: # Outsourcing action
                item_id, outsourcing_choice = poss_actions[idx]
                p, e = graph.items_g2i[item_id]
                if outsourcing_choice == YES:
                    graph, _end_date, _price = outsource_item(graph, instance, item_id, t, enforce_time=False)
                    graph = apply_outsourcing_to_direct_parent(instance, graph, previous_operations, p, e, _end_date)
                    current_cmax = max(current_cmax, _end_date)
                    current_cost = current_cost + _price 
                    DEBUG_PRINT(f"Outsourcing item {item_id} -> ({p},{e})...")
                else:
                    graph.update_item(item_id, [('outsourced', NO)])
                    DEBUG_PRINT(f"Producing item {item_id} -> ({p},{e}) locally...")
                if outsourcing_training_stage:
                    _local_decisions.append(Decision(type=actions_type,
                                                    agent_name=ACTIONS_NAMES[OUTSOURCING],
                                                    target=item_id, 
                                                    value=outsourcing_choice, 
                                                    end_old=old_cmax, 
                                                    end_new=current_cmax, 
                                                    cost_old=old_cost, 
                                                    cost_new=current_cost,
                                                    parent=_local_decisions[-1] if _local_decisions else None, 
                                                    use_cost=True))
            elif actions_type == SCHEDULING: # scheduling action
                operation_id, resource_id = poss_actions[idx]    
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {t}...")     
                graph, utilization, required_types_of_resources, _operation_end = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, utilization, t)
                if instance.simultaneous[p][o]:
                    DEBUG_PRINT("\t >> Simulatenous...")
                    graph, utilization, required_types_of_resources, required_types_of_materials, _operation_end = schedule_other_resources_if_simultaneous(instance, graph, required_types_of_resources, required_types_of_materials, utilization, operation_id, resource_id, p, o, t, _operation_end)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, _operation_end)
                DEBUG_PRINT(f"End of scheduling at time {_operation_end}...")
                current_cmax = max(current_cmax, _operation_end)
                if scheduling_training_stage:
                    _local_decisions.append(Decision(type=actions_type,
                                                    agent_name=ACTIONS_NAMES[SCHEDULING],
                                                    target=operation_id, 
                                                    value=resource_id, 
                                                    end_old=old_cmax, 
                                                    end_new=current_cmax,
                                                    parent=_local_decisions[-1] if _local_decisions else None, 
                                                    use_cost=False))
            else: # Material use action
                operation_id, material_id = poss_actions[idx]
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]}...")  
                graph, required_types_of_materials = apply_use_material(graph, instance, operation_id, material_id, required_types_of_materials, t)
                graph = try_to_open_next_operations(graph, instance, previous_operations, next_operations, operation_id, t)
                current_cmax = max(current_cmax, t)
                if material_use_training_stage and need_reward:
                    _local_decisions.append(Decision(type=actions_type,
                                                    agent_name=ACTIONS_NAMES[MATERIAL_USE],
                                                    target=operation_id,
                                                    value=material_id,
                                                    end_old=current_cmax,
                                                    end_new=current_cmax,
                                                    parent=_local_decisions[-1] if _local_decisions else None, 
                                                    use_cost=False))
            old_cost = current_cost
            old_cmax = current_cmax
        else: # No more possible action at time t
            graph, utilization, t, terminate = manage_current_time(graph, instance, utilization, t)
            if terminate:
                break
    if train:
        if _local_decisions:
            reward_MEMORY.add_or_update_decision(_local_decisions[0], a=alpha, final_cost=current_cost, final_makespan=current_cmax, init_cmax=lb_cmax, init_cost=lb_cost)
            for decision in _local_decisions:
                training_results.add_reward(agent_name=decision.agent_name, reward=decision.reward)  
        return training_results, reward_MEMORY, graph, current_cmax, current_cost
    else:
        return graph, current_cmax, current_cost

# ====================
# =*= V. MAIN CODE =*=
# ====================

def load_trained_models(model_path:str, run_number:int, device:str, fine_tuned: bool = False, size: str = "", id: str = ""):
    index = str(run_number)
    base_name = f"{size}_{id}_" if fine_tuned else ""
    _rm_size = GNN_CONF['resource_and_material_embedding_size']
    _io_size = GNN_CONF['operation_and_item_embedding_size']
    _hidden_size = GNN_CONF['embedding_hidden_channels']
    _ac_size = GNN_CONF['actor_hidden_channels']
    _value_size= GNN_CONF['value_hidden_channels']
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(_rm_size, _io_size, _hidden_size, GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(model_path+'/'+base_name+'gnn_weights_'+index+'.pth', map_location=torch.device(device)))
    shared_critic: L1_CommonCritic = L1_CommonCritic(_rm_size, _io_size, _value_size)
    shared_critic.load_state_dict(torch.load(model_path+'/'+base_name+'critic_weights_'+index+'.pth', map_location=torch.device(device)))
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    scheduling_actor: L1_SchedulingActor = L1_SchedulingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    outsourcing_actor.load_state_dict(torch.load(model_path+'/'+base_name+'outsourcing_weights_'+index+'.pth', map_location=torch.device(device)))
    scheduling_actor.load_state_dict(torch.load(model_path+'/'+base_name+'scheduling_weights_'+index+'.pth', map_location=torch.device(device)))
    material_actor.load_state_dict(torch.load(model_path+'/'+base_name+'material_use_weights_'+index+'.pth', map_location=torch.device(device)))
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

def init_new_models():
    _rm_size = GNN_CONF['resource_and_material_embedding_size']
    _io_size = GNN_CONF['operation_and_item_embedding_size']
    _hidden_size = GNN_CONF['embedding_hidden_channels']
    _ac_size = GNN_CONF['actor_hidden_channels']
    _value_size= GNN_CONF['value_hidden_channels']
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(_rm_size, _io_size, _hidden_size, GNN_CONF['nb_layers'])
    shared_critic: L1_CommonCritic = L1_CommonCritic(_rm_size, _io_size, _value_size)
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    scheduling_actor: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic

def pre_train_on_all_instances(run_number: int, device: str, path: str, debug_mode: bool, interactive: bool = True):
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
    print("Pre-training models with MAPPO (on several instances)...")
    multi_stage_pre_train(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, path=path, solve_function=solve_one, device=device, run_number=run_number, interactive=interactive, debug_mode=debug_mode)

def fine_tune_on_target(id: str, size: str, pre_trained_number: int, path: str, debug_mode: bool, device: str, use_pre_train: bool = False, interactive: bool = True):
    """
        Fine-tune on target instance (size, id)
    """
    agents, shared_embbeding_stack, shared_critic = init_new_models() if not use_pre_train else load_trained_models(model_path=path+directory.models, run_number=pre_trained_number, device=device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for agent,_ in agents:
        agent = agent.to(device)
    print("Fine-tuning models with MAPPO (on target instance)...")
    multi_stage_fine_tuning(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, path=path, solve_function=solve_one, device=device, id=id, size=size, interactive=interactive, debug_mode=debug_mode)

def solve_only_target(id: str, size: str, run_number: int, device: str, path: str, repetitions: int=1):
    """
        Solve the target instance (size, id) only using inference
    """
    target_instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    start_time = systime.time()
    best_cmax = -1.0
    best_cost = -1.0
    best_obj = -1.0
    first = (_run_number<=1)
    agents, shared_embbeding_stack, shared_critic = init_new_models() if first else load_trained_models(model_path=path+directory.models, run_number=run_number, device=device)
    for agent,_ in agents:
        agent = agent.to(device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for rep in range(repetitions):
        print(f"SOLVING INSTANCE {size}_{id} (repetition {rep+1}/{repetitions})...")
        graph, current_cmax, current_cost = solve_one(target_instance, agents, train=False, trainable=[False for _ in agents], device=device, greedy=(rep==0))
        _obj = objective_value(current_cmax, current_cost, target_instance.w_makespan)/100
        if best_obj < 0 or _obj < best_obj:
            best_obj = _obj
            best_cmax = current_cmax
            best_cost = current_cost
    final_metrics = pd.DataFrame({
        'index': [target_instance.id],
        'value': [best_obj],
        'cmax': [best_cmax],
        'cost': [best_cost],
        'repetitions': [repetitions],
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

def solve_all_instances(run_number: int, device: str, path: str):
    """
        Solve all instances only in inference mode
    """
    instances: list[Instance] = load_training_dataset(path=path, train=False)
    for i in instances:
        if (i.size, i.id) not in [('s', 172)]:
            solve_only_target(id=str(i.id), size=str(i.size), run_number=run_number, device=device, path=path, repetitions=SOLVING_REPETITIONS)

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
            # python gns_solver.py --train=true --target=false --mode=prod --number=1 --interactive=true --path=./
            pre_train_on_all_instances(run_number=_run_number, path=args.path, debug_mode=_debug_mode, device=_device, interactive=to_bool(args.interactive))
    else:
        if to_bool(args.target):
            # SOLVE ACTUAL INSTANCE: python gns_solver.py --target=true --size=s --id=151 --train=false --mode=test --path=./ --number=1
            # TRY ON DEBUG INSTANCE: python gns_solver.py --target=true --size=d --id=debug --train=false --mode=test --path=./ --number=1
            solve_only_target(id=args.id, size=args.size, run_number=args.number, device=_device, path=args.path)
        else:
            # python gns_solver.py --train=false --target=false --mode=test --path=./ --number=1
            solve_all_instances(run_number=args.number, device=_device, path=args.path)
    print("===* END OF FILE *===")