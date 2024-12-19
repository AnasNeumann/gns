from model.instance import Instance
from model.solution import HeuristicSolution, Item, Operation, Execution, MaterialUse, Machine, Material
from model.graph import YES, NO, GraphInstance, NeedForResourceFeatures, NeedForMaterialFeatures, ItemFeatures

# =============================================================================
# =*= TRANSLATE GRPAH 2 SOLUTION =*=
# Complete code to translate a model.GraphInstance to a model.HeuristicSolution
# =============================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def min_if_exist(value, old):
    return value if old < 0 else min(value, old)

def translate_solution(graph: GraphInstance, instance: Instance):
    solution: HeuristicSolution = HeuristicSolution()
    solution.start_from_instance(instance)
    
    # 1/4 Outsourcing decisions
    for project in solution.projects:
        for item in project.flat_items:
            item_features: ItemFeatures = ItemFeatures.from_tensor(graph.items()[graph.items_i2g[item.project.id][item.id]], graph.features)
            if item_features.outsourced == YES:
                item.outsourced = True
                item.start = item_features.start_time
                item.end = item_features.end_time
                solution.Cmax = max(solution.Cmax, item_features.end_time)
                solution.total_cost = solution.total_cost + item_features.outsourcing_cost
    
    # 2/4 Execution on finite-capacity resources
    execution_index, execution_features, execution_loop = graph.loop_need_for_resource()
    for i in execution_loop:
        ex_features: NeedForResourceFeatures = NeedForResourceFeatures.from_tensor(execution_features[i], graph.features)
        p, o = graph.operations_g2i[execution_index[0, i].item()]
        r = graph.resources_g2i[execution_index[1, i].item()]
        e = instance.get_item_of_operation(p, o)
        rt = instance.get_resource_type(r)
        operation: Operation = solution.projects[p].flat_operations[o]
        item: Item = operation.item
        execution: Execution = operation.get_machine_usage(rt)
        execution.start = ex_features.start_time
        execution.end = ex_features.end_time
        execution.selected_machine = solution.flat_resources[r]
        item.end = max(item.end, ex_features.end_time) 
        item.start = min_if_exist(ex_features.start_time, item.start)
        solution.Cmax = max(solution.Cmax, ex_features.end_time)

    # 3/4 Execution on consumable materials
    mat_use_index, mat_use_features, mat_use_loop = graph.loop_need_for_material()
    for i in mat_use_loop:
        mat_features: NeedForMaterialFeatures = NeedForMaterialFeatures.from_tensor(mat_use_features[i], graph.features)
        p, o = graph.operations_g2i[mat_use_index[0, i].item()]
        m = graph.materials_g2i[mat_use_index[1, i].item()]
        e = instance.get_item_of_operation(p, o)
        operation: Operation = solution.projects[p].flat_operations[o]
        item: Item = operation.item
        use: MaterialUse = operation.get_material_use(m)
        use.execution_time = mat_features.execution_time
        item.end = max(solution.E_end[p][e], mat_features.execution_time)
        item.start = min_if_exist(mat_features.execution_time, solution.E_start[p][e])
        solution.Cmax = max(solution.Cmax, mat_features.execution_time)

    # 4/4 Build positions
    for res in solution.flat_resources:
        if res.finite_capacity:
            machine: Machine = res
            machine.type.sequence.sort(key=lambda obj: obj.start)
        else:
            material: Material = res
            material.sequence.sort(key=lambda obj: obj.execution_time)
    for rt in solution.machine_types:
        for i, execution in enumerate(rt.sequence):
            execution.position = i
    return solution