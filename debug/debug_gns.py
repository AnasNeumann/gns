from typing import Callable

# =====================================================
# =*= DEBUG THE GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def check_completeness(graph, debug_print: Callable):
    NOT_YET = -1
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

def debug_printer(mode):
    if mode:
        def debug_print(*args):
            print(*args)
            with open('./log.sh', 'a') as file:
                file.write(*args)
                file.write('\n')
    else:
        def debug_print(*_):
            pass
    return debug_print