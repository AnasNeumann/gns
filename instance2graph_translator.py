from model.instance import Instance
from model.graph import GraphInstance, ItemFeatures, OperationFeatures, ResourceFeatures, MaterialFeatures, NeedForMaterialFeatures, NeedForResourceFeatures, YES, NO, NOT_YET

# ====================================================================
# =*= TRANSLATE INSTANCE 2 GRAPH =*=
# Complete code to translate a model.Instance to a model.GraphInstance
# ====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def build_item(i: Instance, graph: GraphInstance, p: int, e: int, head: bool, estimated_start: int, must_be_outsourced: bool=False):
    design_load, physical_load = i.item_processing_time(p, e, total_load=True)
    design_mean_time, physical_mean_time = i.item_processing_time(p, e, total_load=False)
    childrens = i.get_children(p, e, direct=False)
    children_time = 0
    childrens_physical_operations = 0
    childrens_design_operations = 0
    for children in childrens:
        cdt, cpt = i.item_processing_time(p, e=children, total_load=True)
        children_time += (cdt+cpt)
        for child_op in i.loop_item_operations(p, e=children):
            if not i.is_design[p][child_op]:
                childrens_physical_operations += 1
            else:
                childrens_design_operations +=1
    parents_physical_time = 0
    ancestors = i.get_ancestors(p, e)
    for ancestor in ancestors:
        _, apt = i.item_processing_time(p, e=ancestor, total_load=True)
        parents_physical_time += apt
    item_id = graph.add_item(p, e, ItemFeatures(
        head = head,
        external = i.external[p][e],
        outsourced = NOT_YET if i.external[p][e] else NO,
        outsourcing_cost = i.external_cost[p][e],
        outsourcing_time = i.outsourcing_time[p][e],
        remaining_physical_time = physical_load,
        remaining_design_time = design_load,
        parents = len(ancestors),
        children = len(childrens),
        parents_physical_time = parents_physical_time,
        children_time = children_time,
        start_time = estimated_start,
        end_time = -1,
        is_possible = YES if head else NOT_YET))
    start, end = i.get_operations_idx(p,e)
    for o in range(start, end):
        if o > start:
            op_start = op_start+operation_mean_time
        else:
            op_start = estimated_start
        succs = end-(o+1)
        operation_load = i.operation_time(p,o, total_load=True)
        operation_mean_time = i.operation_time(p,o, total_load=False)
        required_res = 0
        required_mat = 0
        for rt in i.required_rt(p, o):
            resources_of_rt = i.resources_by_type(rt)
            if resources_of_rt:
                if i.finite_capacity[resources_of_rt[0]]:
                    required_res += 1
                else:
                    required_mat += 1
            else:
                must_be_outsourced = True
        op_id = graph.add_operation(p, o, OperationFeatures(
            design = i.is_design[p][o],
            sync = i.simultaneous[p][o],
            timescale_hours = i.in_hours[p][o],
            timescale_days = i.in_days[p][o],
            direct_successors = succs,
            total_successors = childrens_physical_operations + succs + (childrens_design_operations if i.is_design[p][o] else 0),
            remaining_time = operation_load,
            remaining_resources = required_res,
            remaining_materials = required_mat,
            available_time = op_start,
            end_time = op_start+operation_mean_time,
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
    estimated_start_child = estimated_start if i.external[p][e] else design_mean_time
    estimated_childrend_end = 0
    estimated_children_cost = 0
    for children in i.get_children(p, e, direct=True):
        graph, child_id, estimated_end_child, child_cost = build_item(i, graph, p, e=children, head=False, estimated_start=estimated_start_child, must_be_outsourced=must_be_outsourced)
        graph.add_item_assembly(item_id, child_id)
        estimated_childrend_end = max(estimated_childrend_end, estimated_end_child)
        estimated_children_cost += child_cost
    external_end = estimated_start + i.outsourcing_time[p][e]
    internal_end = estimated_childrend_end + physical_mean_time
    estimated_end = external_end if must_be_outsourced else min(external_end, internal_end) if i.external[p][e] else internal_end
    graph.update_item(item_id, [('end_time', estimated_end)])
    mandatory_cost = estimated_children_cost + (i.external_cost[p][e] if must_be_outsourced else 0)
    return graph, item_id, estimated_end, mandatory_cost

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

def translate(i: Instance, device: str):
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
    Cmax_lower_bound = 0
    cost_lower_bound = 0
    for p in i.loop_projects():
        graph, _, project_end, project_cost = build_item(i, graph, p, e=i.project_head(p), head=True, estimated_start=0, must_be_outsourced=False)
        Cmax_lower_bound = max(Cmax_lower_bound, project_end)
        cost_lower_bound += project_cost
    graph.operations_i2g = graph.build_i2g_2D(graph.operations_g2i)
    graph.items_i2g = graph.build_i2g_2D(graph.items_g2i)
    graph.add_dummy_item(device=device)
    graph = build_precedence(i, graph)
    graph.to(device)
    return graph, Cmax_lower_bound, cost_lower_bound