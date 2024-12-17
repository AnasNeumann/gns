from model.instance import Instance
from model.solution import HeuristicSolution
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
    solution.random_start_from_instance(instance)

    # Execution on finite-capacity resources
    execution_index, execution_features, execution_loop = graph.loop_need_for_resource()
    for i in execution_loop:
        p, o = graph.operations_g2i[execution_index[0, i].item()]
        r = graph.resources_g2i[execution_index[1, i].item()]
        e = instance.get_item_of_operation(p, o)
        rt = instance.get_resource_type(r)
        solution.selection[p][o][rt] = r
        ex_features: NeedForResourceFeatures = NeedForResourceFeatures.from_tensor(execution_features[i], graph.features)
        solution.O_start[p][o][rt] = ex_features.start_time
        solution.O_end[p][o][rt] = ex_features.end_time
        solution.E_end[p][e] = max(solution.E_end[p][e], ex_features.end_time)
        solution.E_start[p][e] = min_if_exist(ex_features.start_time, solution.E_start[p][e])
        solution.Cmax = max(solution.Cmax, mat_features.end_time)
        if instance.is_design[p][o]:
            solution.E_validated[p][e] = max(solution.E_validated[p][e], ex_features.end_time)
        else:
            solution.E_prod_start[p][e] = min_if_exist(ex_features.start_time, solution.E_prod_start[p][e])

    # Execution on consumable materials
    mat_use_index, mat_use_features, mat_use_loop = graph.loop_need_for_material()
    for i in mat_use_loop:
        p, o = graph.operations_g2i[mat_use_index[0, i].item()]
        m = graph.materials_g2i[mat_use_index[1, i].item()]
        e = instance.get_item_of_operation(p, o)
        rt = instance.get_resource_type(m)
        solution.selection[p][o][rt] = r
        mat_features: NeedForMaterialFeatures = NeedForMaterialFeatures.from_tensor(mat_use_features[i], graph.features)
        solution.O_start[p][o][rt] = mat_features.execution_time
        solution.O_end[p][o][rt] = mat_features.execution_time
        solution.E_end[p][e] = max(solution.E_end[p][e], mat_features.execution_time)
        solution.E_start[p][e] = min_if_exist(mat_features.execution_time, solution.E_start[p][e])
        solution.Cmax = max(solution.Cmax, mat_features.execution_time)
        if instance.is_design[p][o]:
            solution.E_validated[p][e] = max(solution.E_validated[p][e], mat_features.execution_time)
        else:
            solution.E_prod_start[p][e] = min_if_exist(mat_features.execution_time, solution.E_prod_start[p][e])

    # Outsourced items
    for item_id, item in enumerate(graph.items()):
        p, e = graph.items_g2i[item_id]
        item_features: ItemFeatures = ItemFeatures.from_tensor(item, graph.features)
        if item_features.outsourced == YES:
            solution.outsourced[p][e] = True
            solution.E_start[p][e] = item_features.start_time
            solution.E_end[p][e] = solution.E_prod_start[p][e] = solution.E_validated[p][e] = item_features.end_time
            solution.Cmax = max(solution.Cmax, item_features.end_time)
            solution.total_cost = solution.total_cost + item_features.outsourcing_cost 

    '''
        self.sequences = [] # Sequences of each resource rt = (p,o)
        
        self.total_cost = 0
        self.Cmax = 0
        self.E_start, self.outsourced, self.E_prod_start, self.E_validated, self.E_end = [] # Elements (p, e)
        self.O_start, self.O_end = [], [] # Execution of operations (p, o, rt)
        self.selection = [] # Selection of resourses for each operations and feasible resource type p, o, rt = r
    '''
    return solution