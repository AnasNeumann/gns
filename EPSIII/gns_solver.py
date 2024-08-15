import argparse
import pickle
from torch.multiprocessing import Pool, set_start_method
import os
from model import Instance, GraphInstance, L1_ACTOR_CRITIC_GNN, L1_EMBEDDING_GNN
from common import load_instance, to_bool, init_several_1D, search_object_by_id
import torch
torch.autograd.set_detect_anomaly(True)
import random
import multiprocessing

PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
OUTSOURCING = "outsourcing"
SCHEDULING = "scheduling"
MATERIAL_USE = "material_use"
OUTSOURCING_AGENT = 0
SCHEDULING_AGENT = 1
MATERIAL_AGENT = 2
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

def save_models(agents):
    for agent, name in agents:
        torch.save(agent.state_dict(), './models/'+name+'_weights.pth')

def load_trained_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load('./models/gnn_weights.pth'))
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    outsourcing_actor.load_state_dict(torch.load('./models/outsourcing_weights.pth'))
    scheduling_actor.load_state_dict(torch.load('./models/scheduling_weights.pth'))
    material_actor.load_state_dict(torch.load('./models/material_weights.pth'))
    return [(shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')]

def init_new_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    return [(shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')]

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
# =*= SEARCH FOR FEASIBLE ACTIONS =*=
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

# =====================================================
# =*= EXECUTE ONE INSTANCE =*=
# =====================================================

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

def solve_one(instance: Instance, agents, path="", train=False, save=False):
    graph, current_cmax = translate(instance)
    parents = graph.parents()
    related_items = graph.related_items()
    required_types_of_resources, required_types_of_materials, res_by_types = build_required_resources(instance)
    current_time = 0
    current_cost = 0
    rewards, values = torch.Tensor([]), torch.Tensor([])
    probabilities, states, actions, actions_idx = [],[],[],[]
    # TODO Solving / Scheduling algorithm
    actions, actions_type = get_feasible_actions(instance, graph, required_types_of_resources, required_types_of_materials, res_by_types, current_time)
    if train:
        return rewards, values, probabilities, states, actions, actions_idx, [instance.id for _ in rewards], related_items, parents
    else:
        return current_cmax, current_cost 

# ====================================================================
# =*= PPO TRAINING PROCESS FOR MULTI-AGENTS WITH SHARED PARAMETERS =*=
# ====================================================================

def search_instance(instances, id) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

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
    print("\t\t value loss - "+str(value_loss))
    print("\t\t policy loss - "+str(policy_loss)) 
    print("\t\t entropy loss - "+str(entropy_loss)) 
    return (PPO_CONF['policy_loss']*policy_loss) + (PPO_CONF['value_loss']*value_loss) - (entropy_loss*PPO_CONF['entropy'])

def PPO_optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

def async_solve_one(init_args):
    agents, instance = init_args
    total_ops = 0
    for j in instance['jobs']:
        total_ops += len(j)
    print("\t start solving instance: "+str(instance['id'])+"...")
    result = solve_one(instance, agents, train=True)
    print("\t end solving instance: "+str(instance['id'])+"!")
    return result

def async_solve_batch(agents, batch, num_processes, train=True, epochs=-1, optimizers=[]):
    all_probabilities, all_states, all_actions, all_actions_idx, all_instances_idx = init_several_1D(3, [], 5)
    all_related_items, all_parents = [], []
    with Pool(num_processes) as pool:
        results = pool.map(async_solve_one, [(agents, instance) for instance in batch])
    all_rewards, all_values, probabilities, states, actions, actions_idx, instances_idx, related_items, parents = zip(*results)
    for i in range(len(batch)):
        all_parents.append({'id': instances_idx[0][0], 'parents': parents[i]})
        all_related_items.append({'id': instances_idx[0][0], 'related_items': related_items[i]})
        for agent in range(len(agents)):
            all_probabilities[agent].extend(probabilities[i][agent])
            all_states[agent].extend(states[i][agent])
            all_actions[agent].extend(actions[i][agent])
            all_actions_idx[agent].extend(actions_idx[i][agent])
            all_instances_idx[agent].extend(instances_idx[i][agent])
    all_returns = [[ri for r in agent_rewards for ri in calculate_returns(r)] for agent_rewards in all_rewards]
    advantages = []
    flattened_values = []
    for agent in agents:
        advantages.append(torch.Tensor([gae for r, v in zip(all_rewards[agent], all_values[agent]) for gae in generalized_advantage_estimate(r, v)]))
        flattened_values.append([v for vals in all_values[agent] for v in vals])
    if train and epochs>0:
        for e in range(epochs):
            print("\t Optimization epoch: "+str(e+1)+"/"+str(epochs))
            for id, (agent, name) in enumerate(agent):
                print("\t\t Optimizing agent: "+name+"...")
                loss = PPO_loss(batch, agent, all_probabilities[id], all_states[id], all_actions[id], all_actions_idx[id], advantages[id], flattened_values[id], all_returns[id], all_instances_idx[id], all_related_items, all_parents)
                PPO_optimize(optimizers[id], loss)
    else:
        for id, (agent, name) in enumerate(agent):
            print("\t\t Evaluating agent: "+name+"...")
            loss = PPO_loss(batch, agent, all_probabilities[id], all_states[id], all_actions[id], all_actions_idx[id], advantages[id], flattened_values[id], all_returns[id], all_instances_idx[id], all_related_items, all_parents)
            print(f'\t Average Loss = {loss:.4f}')

def train(instances, agents, iterations=PPO_CONF['train_iterations'], batch_size=PPO_CONF['batch_size'], epochs=PPO_CONF['opt_epochs'], validation_rate=PPO_CONF['validation_rate']):
    optimizers = [torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE) for agent,_ in agents]
    for agent,_ in agents:
        agent.train()
        if DEVICE == "cuda":
            agent.to(DEVICE)
    random.shuffle(instances)
    num_val = int(len(instances) * PPO_CONF['validation_ratio'])
    train_instances, val_instances = instances[num_val:], instances[:num_val]
    num_processes = multiprocessing.cpu_count() if PARRALLEL else 1
    print("Running on "+str(num_processes)+" TPUs in parallel...")
    for iteration in range(iterations):
        print("PPO iteration: "+str(iteration+1)+"/"+str(iterations)+":")
        if iteration % PPO_CONF['switch_batch'] == 0:
            print("\t Time to sample new batch of size "+str(batch_size)+"...")
            current_batch = random.sample(train_instances, batch_size)
        async_solve_batch(agents, current_batch, num_processes, train=True, epochs=epochs, optimizers=optimizers)
        if iteration % validation_rate == 0:
            print("\t Time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            with torch.no_grad():
                async_solve_batch(agents, val_instances, num_processes, train=False)
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
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device: {DEVICE}...")
    if to_bool(args.train):
        print("LOAD DATASET TO TRAIN MODELS")
        instances = load_training_dataset()
        print("TRAIN MODELS WITH PPO")
        train(instances, init_new_models())
    else:
        print("SOLVE A TARGET INSTANCE")
        INSTANCE_PATH = './instances/test/'+args.size+'/instance_'+args.id+'.pkl'
        SOLUTION_PATH = './instances/test/'+args.size+'/solution_'+args.id+'.csv'
        instance = load_instance(INSTANCE_PATH)
        solve_one(instance, load_trained_models(), SOLUTION_PATH, train=False, save=True)
    print("===* END OF FILE *===")