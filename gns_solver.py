import argparse
import os
from model.instance import Instance
from model.graph import GraphInstance, State, NO, YES
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
from unistage_pre_training import uni_stage_pre_train as pre_train
from model.agent import MultiAgent_OneInstance
import pickle
from multi_stage_ppo_tuning import multi_stage_fine_tuning
from model.reward_memory import Memory, Transition, Action, Memories
from model.queue import Queue
from torch.optim import Adam

# =====================================================
# =*= 1st MAIN FILE OF THE PROJECT: GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

DEBUG_PRINT: callable = None
LEARNING_RATE = 1e-3
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
SOLVING_REPETITIONS = 10
GNN_CONF = {
    'resource_and_material_embedding_size': 8,
    'operation_and_item_embedding_size': 16,
    'nb_layers': 2,
    'embedding_hidden_channels': 64,
    'value_hidden_channels': 128,
    'actor_hidden_channels': 128}

# =====================================================
# =*= I. SEARCH FOR FEASIBLE ACTIONS =*=
# =====================================================

def can_or_must_outsource_item(instance: Instance, graph: GraphInstance, item_id: int):
    """
        Check if an item can or must be outsourced
    """
    actions = []
    p, e = graph.items_g2i[item_id]
    if graph.item(item_id, 'can_be_outsourced')==YES:
        need_to_be_outsourced = False
        for o in instance.loop_item_operations(p,e):
            for rt in instance.required_rt(p, o):
                if not instance.resources_by_type(rt):
                    DEBUG_PRINT(f"Unavailable resourced {rt} found, Item {item_id} must be outsourced!")
                    need_to_be_outsourced = True
                    break
            if need_to_be_outsourced:
                break
        if need_to_be_outsourced:
            actions.append((item_id, YES))
        else:
            actions.extend([(item_id, YES), (item_id, NO)])
    return actions

def get_outourcing_actions(Q: Queue, instance: Instance, graph: GraphInstance):
    """
        Search possible outsourcing actions
    """
    actions = []
    for item_id in Q.item_queue:
        actions.extend(can_or_must_outsource_item(instance, graph, item_id))
    return actions

def get_scheduling_and_material_use_actions(Q: Queue, instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]]):
    """
        Search possible material use and scheduling actions
    """
    scheduling_actions = []
    material_use_actions = []
    scheduling_execution_times: list[int] = []
    material_execution_times: list[int] = []
    for operation_id in Q.operation_queue:
        p, o = graph.operations_g2i[operation_id]
        available_time = graph.operation(operation_id, 'available_time')
        first_possible_execution_time = next_possible_time(instance, available_time, p, o)
        scheduling_sync_actions = []
        material_sync_actions = []
        if graph.operation(operation_id, 'remaining_resources')>0: # 1. Try for scheduling (and check for sync)
            for rt in required_types_of_resources[p][o]:
                for r in graph.res_by_types[rt]:
                    res_id                        = graph.resources_i2g[r]
                    setup_time                    = compute_setup_time(instance, graph, operation_id, res_id)
                    res_ready_time                = graph.resource(res_id, 'available_time') + setup_time
                    scaled_res_ready_time         = next_possible_time(instance, res_ready_time, p, o)
                    first_possible_execution_time = max(first_possible_execution_time, scaled_res_ready_time)
                    graph.update_need_for_resource(operation_id, res_id, [('setup_time', setup_time)])
                    if not instance.simultaneous[p][o]:
                        scheduling_actions.append((operation_id, res_id))
                        scheduling_execution_times.append(first_possible_execution_time)
                    else:
                        scheduling_sync_actions.append((operation_id, res_id))
        elif graph.operation(operation_id, 'remaining_materials')>0: # 2. Try for material use
            for rt in required_types_of_materials[p][o]:
                for m in graph.res_by_types[rt]:
                    mat_id = graph.materials_i2g[m]
                    mat_possible_time             = available_time if graph.material(mat_id, 'remaining_init_quantity') >= instance.quantity_needed[m][p][o] else max(instance.purchase_time[m], available_time)
                    scaled_mat_possible_time      = next_possible_time(instance, mat_possible_time, p, o)
                    first_possible_execution_time = max(first_possible_execution_time, scaled_mat_possible_time)
                    if not instance.simultaneous[p][o]:
                        material_use_actions.append((operation_id, mat_id))
                        material_execution_times.append(first_possible_execution_time)
                    else:
                        material_sync_actions.append((operation_id, mat_id))
        if scheduling_sync_actions:
                scheduling_actions.extend(scheduling_sync_actions)
                scheduling_execution_times.extend([first_possible_execution_time]*len(scheduling_sync_actions))
        if material_sync_actions:
                material_use_actions.extend(material_sync_actions)
                material_execution_times.extend([first_possible_execution_time]*len(scheduling_sync_actions))
    if scheduling_actions:
        return scheduling_actions, scheduling_execution_times, True
    return material_use_actions, material_execution_times, False

def get_feasible_actions(Q: Queue, instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]]):
    """
        Search next possible actions with priority between decision spaces (outsourcing >> scheduling >> material use)
    """
    actions = [] if not Q.item_queue else get_outourcing_actions(Q, instance, graph)
    type = OUTSOURCING
    execution_times: list[int] = []
    if not actions:
        actions, execution_times, found_scheduling = get_scheduling_and_material_use_actions(Q, instance, graph, required_types_of_resources, required_types_of_materials)
        type = SCHEDULING if found_scheduling else MATERIAL_USE
    return actions, type, execution_times

# =====================================================
# =*= II. APPLY A DECISION MADE =*=
# =====================================================

def outsource_item(Q: Queue, graph: GraphInstance, instance: Instance, item_id: int, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], enforce_time: bool=False, outsourcing_time: int=-1):
    """
        Outsource item and children (reccursive down to the leaves)
    """
    p, e = graph.items_g2i[item_id]
    cost = graph.item(item_id, 'outsourcing_cost')
    outsourcing_start_time = outsourcing_time if enforce_time else graph.item(item_id, 'start_time')
    for child in graph.direct_children[p][e]:
        _, child_end_time, child_cost = outsource_item(Q, graph, instance, graph.items_i2g[p][child], required_types_of_resources, required_types_of_materials, enforce_time=True, outsourcing_time=outsourcing_start_time)
        cost += child_cost
        outsourcing_start_time = max(outsourcing_start_time, child_end_time)
    end_date = outsourcing_start_time + instance.outsourcing_time[p][e]
    graph.update_item(item_id, [
        ('can_be_outsourced', NO),
        ('outsourced', YES),
        ('remaining_time', 0.0),
        ('children_time', 0.0),
        ('start_time', outsourcing_start_time),
        ('end_time', end_date)])
    for o in instance.loop_item_operations(p,e):
        op_id = graph.operations_i2g[p][o]
        if op_id in Q.operation_queue:
            Q.remove_operation(op_id)
        graph.update_operation(op_id, [
            ('remaining_resources', 0.0),
            ('remaining_materials', 0.0),
            ('remaining_time', 0.0)]) 
        for rt in required_types_of_resources[p][o] + required_types_of_materials[p][o]:
            for r in graph.res_by_types[rt]:
                if instance.finite_capacity[r]:
                    res_id = graph.resources_i2g[r]
                    graph.del_need_for_resource(op_id, res_id)
                    graph.inc_resource(res_id, [('remaining_operations', -1)])
                else:
                    mat_id = graph.materials_i2g[r]
                    quantity_needed = graph.need_for_material(op_id, mat_id, 'quantity_needed')
                    graph.del_need_for_material(op_id, mat_id)
                    graph.inc_material(mat_id, [('remaining_demand', -1 * quantity_needed)])
    return outsourcing_start_time, end_date, cost

def apply_outsourcing_to_direct_parent(Q: Queue, instance: Instance, graph: GraphInstance, previous_operations: list, p: int, e: int, end_date: int):
    """
        Apply an outsourcing decision to the direct parent
    """
    for ancestor in graph.ancesors[p][e]:
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [
            ('children_time', graph.item(ancestor_id, 'children_time')-(graph.approximate_design_load[p][e]+graph.approximate_physical_load[p][e]))])
    _parent = graph.direct_parent[p][e]
    for o in graph.first_physical_operations[p][_parent]:
        next_good_to_go: bool = True
        _t = next_possible_time(instance, end_date, p, o)
        graph.update_operation(graph.operations_i2g[p][o], [('available_time', _t)], maxx=True)
        for previous in previous_operations[p][o]:
            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                DEBUG_PRINT(f"\t >> Cannot open parent' first physical operation ({p},{o}) due to ({p},{previous}) not finished! Move at least to {_t}...")
                next_good_to_go = False
                break
        if next_good_to_go:
            DEBUG_PRINT(f"\t >> Opening first physical operation ({p},{o}) of parent {_parent} at {_t}!")
            Q.add_operation(graph.operations_i2g[p][o])

def apply_use_material(graph: GraphInstance, operation_id: int, material_id: int, required_types_of_materials:list[list[list[int]]], use_material_time: int):
    """
        Apply use material to an operation
    """
    p, o             = graph.operations_g2i[operation_id]
    rt               = graph.resource_family[graph.materials_g2i[material_id]]
    quantity_needed  = graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity = graph.material(material_id, 'remaining_init_quantity')
    waiting_demand   = graph.material(material_id, 'remaining_demand') 
    graph.update_need_for_material(operation_id, material_id, [('status', YES), ('execution_time', use_material_time)])
    graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)])
    old_end = graph.operation(operation_id, 'end_time')
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', max(use_material_time, old_end))])
    required_types_of_materials[p][o].remove(rt)

def schedule_operation(graph: GraphInstance, instance: Instance, operation_id: int, resource_id: int, required_types_of_resources: list[list[list[int]]], scheduling_time: int):
    """
        Schedule an operation on a resource
    """
    processing_time = graph.need_for_resource(operation_id, resource_id, 'processing_time')
    p, o = graph.operations_g2i[operation_id]
    operation_end = next_possible_time(instance, scheduling_time + processing_time, p, o)
    e = graph.item_of_operations[p][o]
    r = graph.resources_g2i[resource_id]
    rt = graph.resource_family[r]
    estimated_processing_time = graph.operation_resource_time[p][o][rt]
    item_id = graph.items_i2g[p][e]
    graph.inc_resource(resource_id, [('remaining_operations', -1)])
    graph.update_resource(resource_id, [('available_time', operation_end)])
    graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', scheduling_time),
        ('end_time', operation_end)])
    graph.current_operation_type[resource_id] = instance.get_operation_type(p, o)
    for d in range(instance.nb_settings):
        graph.current_design_value[resource_id][d] == instance.design_value[p][o][d]
    required_types_of_resources[p][o].remove(rt)
    for similar in graph.res_by_types[rt]:
        if similar != r:
            similar_id = graph.resources_i2g[similar]
            graph.inc_resource(similar_id, [('remaining_operations', -1)])
            graph.del_need_for_resource(operation_id, similar_id)
    graph.inc_operation(operation_id, [('remaining_resources', -1), ('remaining_time', -estimated_processing_time)])
    graph.update_operation(operation_id, [('end_time', operation_end), ('started', YES)], maxx=True)
    graph.update_item(item_id, [('start_time', scheduling_time)], minn=True)
    graph.update_item(item_id, [('end_time', operation_end)], maxx=True)
    graph.inc_item(item_id, [('remaining_time', -estimated_processing_time)])
    for ancestor in graph.ancesors[p][e]:
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
    if not instance.is_design[p][o]:
        for child in graph.descendants[p][e]:
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    return operation_end

def schedule_other_resources_if_simultaneous(instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials:list[list[list[int]]], operation_id: int, resource_id: int, p: int, o: int, sync_time: int, operation_end: int):
    """
        Also schedule other resources if the operation is simultaneous
    """
    not_RT: int = graph.resource_family[graph.resources_g2i[resource_id]]
    for rt in required_types_of_resources[p][o] + required_types_of_materials[p][o]:
        if rt != not_RT:
            for r in graph.res_by_types[rt]:
                if instance.finite_capacity[r]:
                    other_resource_id = graph.resources_i2g[r]
                    if graph.resource(other_resource_id, 'available_time') <= sync_time:
                        op_end = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, sync_time)
                        operation_end = max(operation_end, op_end)
                        break
                else:
                    apply_use_material(graph, operation_id, graph.materials_i2g[r], required_types_of_materials, sync_time)
                    break
    return operation_end

def try_to_open_next_operations(Q: Queue, graph: GraphInstance, instance: Instance, previous_operations: list[list[list[int]]], next_operations: list[list[list[int]]], operation_id: int, available_time: int): 
    """
        Try to open next operations after finishing using a resource or material
    """
    p, o = graph.operations_g2i[operation_id]
    e = graph.item_of_operations[p][o]
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
            if graph.is_operation_complete(next_id):
                DEBUG_PRINT("ERRROR: opening a already finished operation!")
            Q.add_operation(next_id)
    if o in graph.last_design_operations[p][e]:
        for child in graph.direct_children[p][e]:
            child_id = graph.items_i2g[p][child]
            if instance.external[p][child]:
                DEBUG_PRINT(f'Enabling item {child_id} -> ({p},{child}) for outsourcing (decision yet to make)...')
                Q.add_item(child_id)
            graph.update_item(child_id, [('start_time', available_time)], maxx=True)

# ====================================================
# =*= III. AUXILIARY FUNCTIONS: BUILD INIT OBJECTS =*=
# ====================================================

def build_required_resources(i: Instance, graph: GraphInstance):
    """
        Build fixed array of required resources per operation
    """
    required_types_of_resources = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    required_types_of_materials = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    res_by_types = [[] for _ in range(i.nb_resource_types)]
    for r in range(i.nb_resources):
        res_by_types[graph.resource_family[r]].append(r)
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

def init_queue(i: Instance, graph: GraphInstance):
    """
        Init the task and time queue
    """
    Q: Queue = Queue()
    for item_id in graph.project_heads:
        p, head = graph.items_g2i[item_id]
        for o in i.first_operations(p, head):
            Q.add_operation(graph.operations_i2g[p][o])
    return Q

# =====================================================
# =*= IV. EXECUTE ONE INSTANCE =*=
# =====================================================

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

def compute_setup_time(instance: Instance, graph: GraphInstance, op_id: int, res_id: int):
    """
        Compute setup times with current design settings and operation types of each finite-capacity resources
    """
    p, o = graph.operations_g2i[op_id]
    r = graph.resources_g2i[res_id]
    op_setup_time = 0 if (instance.get_operation_type(p, o) == graph.current_operation_type[res_id] or graph.current_operation_type[res_id]<0) else instance.operation_setup[r]
    for d in range(instance.nb_settings):
        op_setup_time += 0 if (graph.current_design_value[res_id][d] == instance.design_value[p][o][d] or graph.current_design_value[res_id][d]<0) else instance.design_setup[r][d] 
    return op_setup_time

def next_possible_time(instance: Instance, time_to_test: int, p: int, o: int):
    """
        Search the next possible execution time with correct timescale of the operation
    """
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if time_to_test % scale == 0:
        return time_to_test
    else:
        return ((time_to_test // scale) + 1) * scale

def solve_one(instance: Instance, agents: list[(Module, str)], train: bool, device: str, greedy: bool = False, reward_MEMORY: Memory = None):
    graph, lb_cmax, lb_cost, previous_operations, next_operations, related_items, parent_items = translate(i=instance, device=device)
    required_types_of_resources, required_types_of_materials, graph.res_by_types = build_required_resources(instance, graph)
    alpha: Tensor = torch.tensor([instance.w_makespan], device=device)
    if train:
        _local_decisions: list[Transition] = []
        training_results: MultiAgent_OneInstance = MultiAgent_OneInstance(
            agent_names=ACTIONS_NAMES, 
            instance_id=instance.id,
            related_items=related_items,
            parent_items=parent_items,
            w_makespan=alpha,
            device=device)
    current_cmax = 0
    current_cost = 0
    old_cmax = 0
    old_cost = 0
    DEBUG_PRINT(f"Init Cmax: {lb_cmax} - Init cost: {lb_cost}$")
    Q = init_queue(instance, graph)
    while not Q.done():
        poss_actions, actions_type, execution_times = get_feasible_actions(Q, instance, graph, required_types_of_resources, required_types_of_materials)
        DEBUG_PRINT(f"Current possible {ACTIONS_NAMES[actions_type]} actions: {poss_actions} at times: {execution_times}...")
        if train:
            state: State = graph.to_state(device=device)
            probs, state_value = agents[actions_type][AGENT](state, poss_actions, related_items, parent_items, alpha)
            idx = policy(probs, greedy=False)
            if actions_type != MATERIAL_USE or graph.material(poss_actions[idx][1],'remaining_init_quantity')>0:
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
                _outsourcing_time, _end_date, _price = outsource_item(Q, graph, instance, item_id, required_types_of_resources, required_types_of_materials, enforce_time=False)
                apply_outsourcing_to_direct_parent(Q, instance, graph, previous_operations, p, e, _end_date)
                current_cmax = max(current_cmax, _end_date)
                current_cost = current_cost + _price
                Q.remove_item(item_id)
                DEBUG_PRINT(f"Outsourcing item {item_id} -> ({p},{e}) [start={_outsourcing_time}, end={_end_date}]...")
            else:
                Q.remove_item(item_id)
                graph.update_item(item_id, [('outsourced', NO), ('can_be_outsourced', NO)])
                DEBUG_PRINT(f"Producing item {item_id} -> ({p},{e}) locally...")
            if train:
                _local_decisions.append(Transition(agent_name=ACTIONS_NAMES[OUTSOURCING],
                                                action= Action(type=actions_type, target=item_id, value=outsourcing_choice),
                                                end_old=old_cmax, 
                                                end_new=current_cmax, 
                                                cost_old=old_cost, 
                                                cost_new=current_cost,
                                                parent=_local_decisions[-1] if _local_decisions else None, 
                                                use_cost=True))
        elif actions_type == SCHEDULING: # scheduling action
            operation_id, resource_id = poss_actions[idx]
            p, o = graph.operations_g2i[operation_id]
            DEBUG_PRINT(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {execution_times[idx]}...")     
            _operation_end = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, execution_times[idx])
            if instance.simultaneous[p][o]:
                DEBUG_PRINT("\t >> Simulatenous...")
                _operation_end = schedule_other_resources_if_simultaneous(instance, graph, required_types_of_resources, required_types_of_materials, operation_id, resource_id, p, o, execution_times[idx], _operation_end)
            if graph.is_operation_complete(operation_id):
                Q.remove_operation(operation_id)
                try_to_open_next_operations(Q, graph, instance, previous_operations, next_operations, operation_id, _operation_end)
            DEBUG_PRINT(f"End of scheduling at time {_operation_end}...")
            current_cmax = max(current_cmax, _operation_end)
            if train:
                _local_decisions.append(Transition(agent_name=ACTIONS_NAMES[SCHEDULING],
                                                action= Action(type=actions_type, target=operation_id, value=resource_id),
                                                end_old=old_cmax, 
                                                end_new=current_cmax,
                                                parent=_local_decisions[-1] if _local_decisions else None, 
                                                use_cost=False))
        else: # Material use action
            operation_id, material_id = poss_actions[idx]
            p, o = graph.operations_g2i[operation_id]
            DEBUG_PRINT(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]} at time {execution_times[idx]}...")  
            apply_use_material(graph, operation_id, material_id, required_types_of_materials, execution_times[idx])
            if graph.is_operation_complete(operation_id):
                Q.remove_operation(operation_id)
                try_to_open_next_operations(Q, graph, instance, previous_operations, next_operations, operation_id, execution_times[idx])
            current_cmax = max(current_cmax, execution_times[idx])
            if train and need_reward:
                _local_decisions.append(Transition(agent_name=ACTIONS_NAMES[MATERIAL_USE],
                                                action= Action(type=actions_type, target=operation_id, value=material_id),
                                                end_old=old_cmax,
                                                end_new=current_cmax,
                                                parent=_local_decisions[-1] if _local_decisions else None, 
                                                use_cost=False))
        old_cost = current_cost
        old_cmax = current_cmax
    if train:
        reward_MEMORY.add_or_update_decision(_local_decisions[0], a=alpha, final_cost=current_cost, final_makespan=current_cmax, init_cmax=lb_cmax, init_cost=lb_cost)
        for decision in _local_decisions:
            training_results.add_reward(agent_name=decision.agent_name, reward=decision.reward)  
        return training_results, reward_MEMORY, graph, current_cmax, current_cost
    else:
        return graph, current_cmax, current_cost

# ====================
# =*= V. MAIN CODE =*=
# ====================

SOLVING_SIZES: list[str] = ['s']
def load_dataset(path: str, train: bool = True):
    type: str = '/train/' if train else '/test/'
    instances = []
    for size in SOLVING_SIZES:
        complete_path = path+directory.instances+type+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def load_trained_models(model_path:str, run_number:int, device:str, fine_tuned: bool = False, size: str = "", id: str = "", training_stage: bool=True):
    index = str(run_number)
    base_name = f"{size}_{id}_" if fine_tuned else ""
    _rm_size = GNN_CONF['resource_and_material_embedding_size']
    _io_size = GNN_CONF['operation_and_item_embedding_size']
    _hidden_size = GNN_CONF['embedding_hidden_channels']
    _ac_size = GNN_CONF['actor_hidden_channels']
    _value_size= GNN_CONF['value_hidden_channels']
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(_rm_size, _io_size, _hidden_size, GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(model_path+'/'+base_name+'gnn_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
    shared_critic: L1_CommonCritic = L1_CommonCritic(_rm_size, _io_size, _value_size)
    shared_critic.load_state_dict(torch.load(model_path+'/'+base_name+'critic_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    scheduling_actor: L1_SchedulingActor = L1_SchedulingActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, shared_critic, _rm_size, _io_size, _ac_size)
    outsourcing_actor.load_state_dict(torch.load(model_path+'/'+base_name+'outsourcing_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
    scheduling_actor.load_state_dict(torch.load(model_path+'/'+base_name+'scheduling_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
    material_actor.load_state_dict(torch.load(model_path+'/'+base_name+'material_use_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
    shared_GNN = shared_GNN.to(device)
    shared_critic = shared_critic.to(device)
    outsourcing_actor = outsourcing_actor.to(device)
    material_actor = material_actor.to(device)
    scheduling_actor = scheduling_actor.to(device)
    outsourcing_actor.train()
    scheduling_actor.train()
    material_actor.train()
    torch.compile(outsourcing_actor)
    torch.compile(scheduling_actor)
    torch.compile(material_actor)
    if training_stage:
        optimizer = Adam(list(scheduling_actor.parameters()) + list(material_actor.parameters()) + list(outsourcing_actor.parameters()), lr=LEARNING_RATE)
        optimizer.load_state_dict(torch.load(model_path+'/'+base_name+'adam_weights_'+index+'.pth', map_location=torch.device(device), weights_only=True))
        with open(model_path+'/'+base_name+'memory_'+index+'.pth', 'rb') as file:
            memory: Memories = pickle.load(file)
        return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic, optimizer, memory
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])]

def init_new_models(device: str, training_stage: bool=True):
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
    shared_GNN = shared_GNN.to(device)
    shared_critic = shared_critic.to(device)
    outsourcing_actor = outsourcing_actor.to(device)
    material_actor = material_actor.to(device)
    scheduling_actor = scheduling_actor.to(device)
    outsourcing_actor.train()
    scheduling_actor.train()
    material_actor.train()
    torch.compile(outsourcing_actor)
    torch.compile(scheduling_actor)
    torch.compile(material_actor)
    if training_stage:
        optimizer = Adam(list(scheduling_actor.parameters()) + list(material_actor.parameters()) + list(outsourcing_actor.parameters()), lr=LEARNING_RATE)
        memory: Memories = Memories()
        return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])], shared_GNN, shared_critic, optimizer, memory
    return [(outsourcing_actor, ACTIONS_NAMES[OUTSOURCING]), (scheduling_actor, ACTIONS_NAMES[SCHEDULING]), (material_actor, ACTIONS_NAMES[MATERIAL_USE])]

def pre_train_on_all_instances(run_number: int, device: str, path: str):
    """
        Pre-train networks on all instances
    """
    first = (_run_number<=1)
    previous_run = run_number - 1
    agents, shared_embbeding_stack, shared_critic, optimizer, memory = init_new_models(device=device) if first else load_trained_models(model_path=path+directory.models, run_number=previous_run, device=device)
    print("Pre-training models with MAPPO (on several instances)...")
    pre_train(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, optimizer=optimizer, memory=memory, path=path, solve_function=solve_one, device=device, run_number=run_number)

def fine_tune_on_target(id: str, size: str, pre_trained_number: int, path: str, debug_mode: bool, device: str, use_pre_train: bool = False, interactive: bool = True):
    """
        Fine-tune on target instance (size, id)
    """
    agents, shared_embbeding_stack, shared_critic, _, _ = init_new_models(device=device) if not use_pre_train else load_trained_models(model_path=path+directory.models, run_number=pre_trained_number, device=device)
    shared_embbeding_stack = shared_embbeding_stack.to(device)
    shared_critic = shared_critic.to(device)
    for agent,_ in agents:
        agent = agent.to(device)
    print("Fine-tuning models with MAPPO (on target instance)...")
    multi_stage_fine_tuning(agents=agents, embedding_stack=shared_embbeding_stack, shared_critic=shared_critic, path=path, solve_function=solve_one, device=device, id=id, size=size, interactive=interactive, debug_mode=debug_mode)

def solve_only_target(id: str, size: str, agents: list[(str, Module)], run_number: int, device: str, path: str, repetitions: int=1):
    """
        Solve the target instance (size, id) only using inference
    """
    target_instance: Instance = load_instance(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    DEBUG_PRINT(target_instance.display())
    start_time = systime.time()
    best_cmax = -1.0
    best_cost = -1.0
    best_obj = -1.0
    for rep in range(repetitions):
        print(f"SOLVING INSTANCE {size}_{id} (repetition {rep+1}/{repetitions})...")
        graph, current_cmax, current_cost = solve_one(target_instance, agents, train=False, device=device, greedy=(rep==0))
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
    return target_instance, solution

def solve_all_instances(agents: list[(str, Module)], run_number: int, device: str, path: str):
    """
        Solve all instances only in inference mode
    """
    instances: list[Instance] = load_dataset(path=path, train=False)
    for i in instances:
        if (i.size, i.id) not in [('s', 172)]:
            solve_only_target(id=str(i.id), size=str(i.size), agents=agents, run_number=run_number, device=device, path=path, repetitions=SOLVING_REPETITIONS)

def agents_ready(device: str, run_number: int, path: str):
    first = (run_number<=1)
    agents = init_new_models(device=device, training_stage=False) if first else load_trained_models(model_path=path+directory.models, run_number=run_number, device=device, training_stage=False)
    for agent,_ in agents:
        agent = agent.to(device)
    return agents

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
            pre_train_on_all_instances(run_number=_run_number, path=args.path, device=_device)
    else:
        agents: list[(str, Module)] = agents_ready(device=_device, run_number=_run_number, path=args.path)
        if to_bool(args.target):
            # SOLVE ACTUAL INSTANCE: python gns_solver.py --target=true --size=xxl --id=151 --train=false --mode=test --path=./ --number=1
            # TRY ON DEBUG INSTANCE: python gns_solver.py --target=true --size=d --id=debug --train=false --mode=test --path=./ --number=1
            i, s = solve_only_target(id=args.id, size=args.size,agents=agents, run_number=args.number, device=_device, path=args.path)
        else:
            # python gns_solver.py --train=false --target=false --mode=prod --path=./ --number=1
            solve_all_instances(run_number=args.number, agents=agents, device=_device, path=args.path)
    print("===* END OF FILE *===")