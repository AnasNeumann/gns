import argparse
from tools.common import init_several_1D, init_2D, init_several_2D, init_3D, load_instance, directory
from model.instance import Instance
from model.solution import Solution, HeuristicSolution, Item, Operation
import pandas as pd
from ortools.sat.python import cp_model
import time as systime

# ###############################################
# =*= EXACT CP Solver (Using Google OR-Tools) =*=
# ###############################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

BASIC_PATH = './'

def init_vars(model: cp_model.CpModel, i: Instance):
    s                = Solution()
    nb_projects      = i.get_nb_projects()
    elts_per_project = i.E_size[0]
    s.Cmax           = model.NewIntVar(0, i.M, 'Cmax')
    s.E_start        = [[model.NewIntVar(0, i.M, f'E_start_{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_outsourced   = [[model.NewBoolVar(f'E_outsource{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_prod_start   = [[model.NewIntVar(0, i.M, f'E_prod_start{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_end          = [[model.NewIntVar(0, i.M, f'E_end{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_validated    = [[model.NewIntVar(0, i.M, f'E_validated{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.precedes       = init_2D(nb_projects, nb_projects, None)
    s.O_uses_init_quantity, s.O_start, s.O_setup, s.O_end, s.O_executed, s.D_setup = init_several_1D(nb_projects, None, 6)
    for p in range(nb_projects):
        nb_ops       = i.O_size[p]
        s.D_setup[p] = init_3D(nb_ops, i.nb_resources, i.nb_settings, None)
        s.O_uses_init_quantity[p], s.O_start[p], s.O_setup[p], s.O_end[p], s.O_executed[p] = init_several_2D(nb_ops, i.nb_resources, None, 5)
        for o in range(nb_ops):
            for r in i.required_resources(p, o):
                s.O_uses_init_quantity[p][o][r] = model.NewBoolVar(f'O_uses_init_quantity{p}_{o}_{r}')
                s.O_start[p][o][r]              = model.NewIntVar(0, i.M, f'O_start{p}_{o}_{r}')
                s.O_setup[p][o][r]              = model.NewBoolVar(f'O_setup{p}_{o}_{r}')
                s.O_end[p][o][r]                = model.NewIntVar(0, i.M, f'O_end{p}_{o}_{r}')
                s.O_executed[p][o][r]           = model.NewBoolVar(f'O_executed{p}_{o}_{r}')
                s.D_setup[p][o][r]              = [model.NewBoolVar(f'D_setup{p}_{o}_{r}_{c}') for c in range(i.nb_settings)]
        for p2 in range(nb_projects):
            nb_ops2 = i.O_size[p2]
            s.precedes[p][p2] = init_3D(nb_ops, nb_ops2, i.nb_resources, None)
            for o in range(nb_ops):
                for o2 in range(nb_ops2):
                    for r in i.required_resources(p, o):
                        if i.require(p2, o2, r):
                            s.precedes[p][p2][o][o2][r] = model.NewBoolVar(f'precedes{p}_{p2}_{o}_{o2}_{r}')
    return model, s

def init_objective_function(model: cp_model.CpModel, i: Instance, s: Solution):
    weight = int(100 * i.w_makespan)
    s.obj.append(s.Cmax * weight)
    for p in i.loop_projects():
        for e in i.loop_items(p):
            if i.external[p][e]:
                s.obj.append(s.E_outsourced[p][e] * i.external_cost[p][e] * (100 - weight))
    model.Minimize(sum(s.obj))
    return model, s

# Cmax computation: after the end of the head item of all projects
def c1(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        model.Add(s.Cmax >= s.E_end[p][i.project_head(p)])
    return model, s

# End of non-outsourced item (after the end of the last operation)
def c2(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            for o in i.last_operations(p, e):
                for r in i.required_resources(p, o):
                    model.Add(s.E_end[p][e] + (-i.M)*s.O_executed[p][o][r] + (-i.real_time_scale(p,o))*s.O_end[p][o][r] >= -1 * i.M)
    return model, s

# End of outsourced item (based on start and outsourcing_time)
def c3(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            model.Add(s.E_end[p][e] + (-1)*s.E_prod_start[p][e] + (-i.M)*s.E_outsourced[p][e] >= i.outsourcing_time[p][e] + -1 * i.M)
    return model, s

# Physical start of non-outsourced item (before start of first physical operation. 
# The start time is multiplied by the right timescale (for example: design is in days; assembly in hours; and minutes in minutes)
def c4(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                if not i.is_design[p][o]:
                    for r in i.required_resources(p, o):
                        model.Add(s.E_prod_start[p][e] + i.M*s.O_executed[p][o][r] + (-i.real_time_scale(p,o))*s.O_start[p][o][r] <= i.M)
    return model, s

# Subcontract only if possible (known supplier)
def c5(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            if not i.external[p][e]:
                model.Add(s.E_outsourced[p][e] <= 0)
    return model, s

# Subcontract a whole branch of elements: if a prent item is outsourced, its children too!
def c6(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            for e2 in i.get_children(p, e, True):
                model.Add(s.E_outsourced[p][e2] + (-1)*s.E_outsourced[p][e] >= 0)
    return model, s

# Available quantity of raw material (resource with finite-capacity = False) before purchase: we can't schedule more operations that possible!
def c7(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_consumable_materials():
        terms = []
        for p, o in i.operations_by_resource(r):
            terms.append(i.quantity_needed[r][p][o]*s.O_uses_init_quantity[p][o][r])
        if len(terms)>0:
            model.Add(sum(terms) <= i.init_quantity[r])
    return model, s

# Answer to the question: is an operation executed with the initial quantity (hence, before the purchase) or needs to wait for purchase_time?
def c8(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r in i.required_resources(p, o):
                if not i.finite_capacity[r]:
                    model.Add(i.M*s.O_uses_init_quantity[p][o][r] + (-i.M)*s.O_executed[p][o][r] + i.real_time_scale(p,o)*s.O_start[p][o][r] >= i.purchase_time[r] + (-1)*i.M)
    return model, s

# Complete execution of an operation on all required types of resources (if the related item not outsourced): one selected by required type of resource!
def c9(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                for rt in i.required_rt(p, o):
                    terms = []
                    for r in i.resources_by_type(rt):
                        terms.append(s.O_executed[p][o][r])
                    model.Add(s.E_outsourced[p][e] + sum(terms) == 1)
    return model, s

# Simultaneous operations (sync of resources mandatory) : only execute when all needed resources are available!
def c10(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            if i.simultaneous[p][o]:
                for r in i.required_resources(p, o):
                    for v in i.required_resources(p, o):
                        if r != v:
                            model.Add(s.O_start[p][o][r] + (-1)*s.O_start[p][o][v] + i.M*s.O_executed[p][o][r] + i.M*s.O_executed[p][o][v] <= 2*i.M)
    return model, s

# End of an operation according to the execution time and possible start time (start time is measured in the right timescale and multiplied to obtain the value in minutes)
def c11(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r in i.required_resources(p, o):
                if i.finite_capacity[r]:
                    model.Add(i.real_time_scale(p,o)*s.O_end[p][o][r] + (-i.real_time_scale(p,o))*s.O_start[p][o][r] + (-i.M)*s.O_executed[p][o][r] >= i.execution_time[r][p][o] + (-i.M))
    return model, s

# Precedence relations between operations of one same element and execution time on choosen resources (executed)
# o2 must start after the end of o1 since its his successor (start time is measured in the right timescale and multiplied to obtain the value in minutes)
def c12(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o1 in range(start, end):
                for o2 in range(start, end):
                    if o1 != o2 and i.precedence[p][e][o1][o2]:
                        for r in i.required_resources(p, o1):
                            for v in i.required_resources(p, o2):
                                model.Add(i.real_time_scale(p,o1)*s.O_start[p][o1][r] + (-i.real_time_scale(p,o2))*s.O_end[p][o2][v] + (-i.M)*s.O_executed[p][o1][r] + (-i.M)*s.O_executed[p][o2][v] >= (-2)*i.M)
    return model, s

# Start time of parent' production (first physical operations or outsourcing time) is right after the end of its children 
# This is also true for outsourced items!
def c13(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.get_children(p, e1, direct=True):
                model.Add(s.E_prod_start[p][e1] + (-1)*s.E_end[p][e2] >= 0)
    return model, s

# Production start time (before outsourcing or first physical operation) must be after design validation (end of its last design operation)
def c14(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            model.Add(s.E_prod_start[p][e] + (-1)*s.E_validated[p][e] >= 0)
    return model, s

# Design validation must be after parent' validation (end of its last design operation)
def c15(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.get_children(p, e1, direct=True):
                model.Add(s.E_validated[p][e2] + (-s.E_validated[p][e1]) >= 0)
    return model, s

# Start of any operation (design or physical: assembly or production) only after parent' validation (end of its last design operation)
def c16(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.get_children(p, e1, direct=True):
                start, end = i.get_operations_idx(p, e2)
                for o in range(start, end):
                    for r in i.required_resources(p, o):
                        model.Add(i.real_time_scale(p,o)*s.O_start[p][o][r] + (-1)*s.E_validated[p][e1] + (-i.M)*s.O_executed[p][o][r] >= (-1)*i.M)
    return model, s

# No more than one direct successor (by required finite-capacity resource)
def c17(model: cp_model.CpModel, i: Instance, s: Solution):
    for p1 in i.loop_projects():
        for o1 in i.loop_operations(p1):
            for r in i.required_resources(p1,o1):
                if i.finite_capacity[r]:
                    terms = []
                    for p2, o2 in i.operations_by_resource(r):
                        if not i.is_same(p1,p2,o1,o2):
                            terms.append(s.precedes[p1][p2][o1][o2][r])
                    if len(terms)>0:
                        model.Add(sum(terms) <= 1)
    return model, s

# No more than one direct predecessor (by required finite-capacity resource)
def c18(model: cp_model.CpModel, i: Instance, s: Solution):
    for p1 in i.loop_projects():
        for o1 in i.loop_operations(p1):
            for r in i.required_resources(p1,o1):
                if i.finite_capacity[r]:
                    terms = []
                    for p2, o2 in i.operations_by_resource(r):
                        if not i.is_same(p1,p2,o1,o2):
                            terms.append(s.precedes[p2][p1][o2][o1][r])
                    if len(terms)>0:
                        model.Add(sum(terms) <= 1)
    return model, s

# No operation can be its own successor or predecessor on finite-capacity resources!
def c19(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        terms = []
        for p, o in i.operations_by_resource(r):
            terms.append(s.precedes[p][p][o][o][r])
        if len(terms)>0:
            model.Add(sum(terms) == 0)
    return model, s

# Total number of operations in a finite-capacity resource: as much executed operation as there is precedence relations
def c20(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        terms_pos = []
        terms_neg = []
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2):
                    terms_pos.append(s.precedes[p1][p2][o1][o2][r])
        for p, o in i.operations_by_resource(r):
            terms_neg.append(s.O_executed[p][o][r])
        if len(terms_pos)>0 or len(terms_neg)>0:
            model.Add(sum(terms_pos) + (-1)*sum(terms_neg)>= -1)
    return model, s

# Precedence only for operations executed by the resource (especially if several similar resources with same RT are available)
def c21(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2):
                    model.Add(s.O_executed[p1][o1][r] + (-1)*s.precedes[p1][p2][o1][o2][r] >= 0)                     
    return model, s

# Precedence only for operations executed by the resource (same constraint as c21 but the other way)
def c22(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2):
                    model.Add(s.O_executed[p2][o2][r] + (-1)*s.precedes[p1][p2][o1][o2][r] >= 0) 
    return model, s

# Start operation only after the end of its predecessor (by resource) considering operation_setups, all possible design_setup, predecessor end time, and right timescale
def c23(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2):
                    terms = []
                    for c in range(i.nb_settings):
                        terms.append(i.design_setup[r][c] * s.D_setup[p1][o1][r][c])
                    model.Add(i.real_time_scale(p1,o1)*s.O_start[p1][o1][r] + (-1)*sum(terms) + (-i.real_time_scale(p2,o2))*s.O_end[p2][o2][r] + (-i.operation_setup[r])*s.O_setup[p1][o1][r] + (-i.M)*s.precedes[p2][p1][o2][o1][r]>= (-1) * i.M)
    return model, s

# Operation setups: with two consecutive operations on the same resources don't share the same type (OT)  
def c24(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2) and i.get_operation_type(p1,o1) != i.get_operation_type(p2, o2):
                    model.Add(s.precedes[p2][p1][o2][o1][r] + (-1)*s.O_setup[p1][o1][r] <= 0)
    return model, s

# Setups for design parameters: with two consecutive operations on the same resources don't share the same design values on different settings  
def c25(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in i.get_finie_capacity_resources():
        for p1, o1 in i.operations_by_resource(r):
            for p2, o2 in i.operations_by_resource(r):
                if not i.is_same(p1,p2,o1,o2):
                    for c in range(i.nb_settings):
                        if i.design_value[p1][o1][c] != i.design_value[p2][o2][c]:
                            model.Add(s.precedes[p2][p1][o2][o1][r] + (-1)*s.D_setup[p1][o1][r][c] <= 0)
    return model, s

# Validation of an element: after the end of its last design operations
def c26(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                if i.is_design[p][o]:
                    for r in i.required_resources(p,o):
                        model.Add(s.E_validated[p][e] + (-i.M)*s.O_executed[p][o][r] + (-i.real_time_scale(p,o))*s.O_end[p][o][r] >= (-1) * i.M)
    return model, s

# No double execution on the same type of resources (by operation): only choose one the available resources
def c27(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r1 in i.required_resources(p,o):
                for r2 in i.required_resources(p,o):
                    if r1 != r2 and i.get_resource_familly(r1) == i.get_resource_familly(r2):
                        model.Add(s.O_executed[p][o][r1] + s.O_executed[p][o][r2] <= 1)
    return model, s

# ##############################
# =*= FEASIBILITY CHECK ONLY =*=
# ##############################

def derive_production_start(item: Item) -> int:
    phys = [exe.start for op in item.production_ops for exe in op.machine_usage]
    result = min(phys) if phys else int(item.start)
    return int(result[0]) if isinstance(result, tuple) else int(result)

def derive_validation(item: Item) -> int:
    des = [exe.end for op in item.design_ops for exe in op.machine_usage]
    result = max(des) if des else int(item.start)
    return int(result[0]) if isinstance(result, tuple) else int(result)

# Fixing the variables according to the solution (for checking only)
def fix_item_vars(model: cp_model.CpModel, s: Solution, sol: HeuristicSolution):
    for proj in sol.projects:
        p = proj.id
        for item in proj.flat_items:
            e = item.id
            model.Add(s.E_prod_start[p][e] == derive_production_start(item))
            model.Add(s.E_validated [p][e] == derive_validation(item))
            model.Add(s.E_outsourced[p][e] == int(item.outsourced))
            model.Add(s.E_start[p][e]      == int(item.start))
            model.Add(s.E_end[p][e]        == int(item.end))

# Fixing operation variables according to the solution (for checking only)
def fix_operation_vars(model: cp_model.CpModel, i: Instance, s: Solution, sol: HeuristicSolution):
    for proj in sol.projects:
        p = proj.id
        for op in proj.flat_operations:
            o = op.id
            scale = i.real_time_scale(p,o)
            # --- finite‑capacity resources --------------------------------
            for exe in op.machine_usage:
                r  = exe.selected_machine.id
                _start = exe.start[0] if isinstance(exe.start, tuple) else exe.start
                _end = exe.end[0] if isinstance(exe.end, tuple) else exe.end
                model.Add(s.O_executed[p][o][r] == 1)
                model.Add(s.O_start   [p][o][r] == int(_start // scale))
                model.Add(s.O_end     [p][o][r] == int(_end // scale))
            for r in i.required_resources(p,o):
                if i.finite_capacity[r] and \
                   op.get_machine_usage(i.get_resource_type(r)) is None:
                    model.Add(s.O_executed[p][o][r] == 0)
            # --- consumables ---------------------------------------------
            for use in op.material_use:
                m = use.material.id
                model.Add(s.O_uses_init_quantity[p][o][m] == int(use.execution_time < i.purchase_time[m]))

# fixing the setup variables according to the solution (for checking only)
def fix_precedes_and_setups(model: cp_model.CpModel, i: Instance, s: Solution, sol: HeuristicSolution):
    for rt in sol.machine_types:
        for seq in [rt.sequence]:
            for idxA in range(len(seq)-1):
                a = seq[idxA];  b = seq[idxA+1]
                pA,oA = a.operation.item.project.id, a.operation.id
                pB,oB = b.operation.item.project.id, b.operation.id
                r     = a.selected_machine.id
                model.Add(s.precedes[pA][pB][oA][oB][r] == 1)
                model.Add(s.O_setup [pB][oB][r] == int(a.operation.operation_family != b.operation.operation_family))
                for d,val in enumerate(i.design_value[pB][oB]):
                    diff = int(val != a.operation.design_value[d])
                    model.Add(s.D_setup[pB][oB][r][d] == diff)
            if seq:
                first = seq[0]
                p,o = first.operation.item.project.id, first.operation.id
                r   = first.selected_machine.id
                model.Add(sum(s.precedes[p1][p][o1][o][r] for p1 in i.loop_projects() for o1 in i.loop_operations(p1) if i.require(p1,o1,r) and not (p1==p and o1==o)) == 0)

def CHECK_FEASIBILITY(model: cp_model.CpModel, i: Instance, s: Solution, sol: HeuristicSolution):
    model.Add(s.Cmax == int(sol.Cmax))
    fix_item_vars(model, s, sol)
    fix_operation_vars(model, i, s, sol)
    fix_precedes_and_setups(model, i, s, sol)

# ###############################################

def solve_one(instance: Instance, cpus: int, memory: int, time: int, solution_path: str):
    start_time = systime.time()
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time * 60.0 * 60.0
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.num_search_workers = cpus
    solver.parameters.max_memory_in_mb = memory
    solver.parameters.absolute_gap_limit = 5.0 
    solver.parameters.use_implied_bounds = True
    solver.parameters.use_probing_search = True
    solver.parameters.cp_model_presolve = True
    solver.parameters.optimize_with_core = True
    solver.parameters.log_search_progress = True
    solver.parameters.enumerate_all_solutions = False
    model, solution = init_vars(model, instance)
    model, solution = init_objective_function(model, instance, solution)
    for constraint in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27]:
        model, solution = constraint(model, instance, solution)
    status = solver.Solve(model)
    computing_time = systime.time()-start_time
    if status == cp_model.OPTIMAL:
        solutions_df = pd.DataFrame({'index': [instance.id], 'value': [solver.ObjectiveValue()/100], 'gap': [0], 'status': ['optimal'], 'computing_time': [computing_time], 'max_time': [time], 'cpu': [cpus], 'max_memory': [memory]})
    elif status == cp_model.FEASIBLE:
        best_objective = solver.ObjectiveValue()
        lower_bound = solver.BestObjectiveBound()
        gap = abs(best_objective - lower_bound) / abs(best_objective) if best_objective != 0 else -1
        solutions_df = pd.DataFrame({'index': [instance.id], 'value': [best_objective/100], 'gap': [gap], 'status': ['feasible'], 'computing_time': [computing_time], 'max_time': [time], 'cpu': [cpus], 'max_memory': [memory]})
    else:
        solutions_df = pd.DataFrame({'index': [instance.id], 'value': [-1], 'status': ['failure'], 'computing_time': [computing_time], 'max_time': [time], 'cpu': [cpus], 'max_memory': [memory]})
    print(solutions_df)
    solutions_df.to_csv(solution_path, index=False)

'''
    TEST WITH
    python exact_solver.py --size=s --id=151 --mode=test --path=./ --time=1 --memory=8
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=True)
    parser.add_argument("--id", help="Id of the solved instance", required=True)
    parser.add_argument("--mode", help="Execution mode", required=True)
    parser.add_argument("--memory", help="Execution max memory", required=True)
    parser.add_argument("--time", help="Max computing time", required=True)
    parser.add_argument("--path", help="Saving path on the server", required=True)
    args = parser.parse_args()
    BASIC_PATH = args.path
    cpus = 8 if args.mode == 'test' else 32
    time = int(args.time)
    memory = int(args.memory)
    print(f'CPU USED: {cpus}')
    INSTANCE_PATH = BASIC_PATH+directory.instances+'/test/'+args.size+'/instance_'+args.id+'.pkl'
    SOLUTION_PATH = BASIC_PATH+directory.instances+'/test/'+args.size+'/solution_exact_'+args.id+'.csv'
    print(f"Loading {INSTANCE_PATH}...")
    instance = load_instance(INSTANCE_PATH)
    print("===* START SOLVING *===")
    solve_one(instance, cpus, memory, time, SOLUTION_PATH)
    print("===* END OF FILE *===")