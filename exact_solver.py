import argparse
from tools.common import init_several_1D, init_2D, init_several_2D, init_3D, load_instance, directory
from model.instance import Instance
from model.solution import Solution
import pandas as pd
from ortools.sat.python import cp_model
import time as systime

# =====================================================
# =*= EXACT CP Solver (Using Google OR-Tools) =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

BASIC_PATH = './'

def init_vars(model: cp_model.CpModel, i: Instance):
    s = Solution()
    nb_projects = i.get_nb_projects()
    elts_per_project = i.E_size[0]
    s.E_start = [[model.NewIntVar(0, i.M, f'E_start_{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_outsourced = [[model.NewBoolVar(f'E_outsource{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_prod_start = [[model.NewIntVar(0, i.M, f'E_prod_start{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_end = [[model.NewIntVar(0, i.M, f'E_end{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_validated = [[model.NewIntVar(0, i.M, f'E_validated{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.O_uses_init_quantity, s.O_start, s.O_setup, s.O_end, s.O_executed, s.D_setup = init_several_1D(nb_projects, None, 6)
    s.precedes = init_2D(nb_projects, nb_projects, None)
    s.Cmax = model.NewIntVar(0, i.M, 'Cmax')
    for p in range(nb_projects):
        nb_ops = i.O_size[p]
        s.O_uses_init_quantity[p], s.O_start[p], s.O_setup[p], s.O_end[p], s.O_executed[p] = init_several_2D(nb_ops, i.nb_resources, None, 5)
        s.D_setup[p] = init_3D(nb_ops, i.nb_resources, i.nb_settings, None)
        for o in range(nb_ops):
            for r in range(i.nb_resources):
                if i.require(p, o, r):
                    s.O_uses_init_quantity[p][o][r] = model.NewBoolVar(f'O_uses_init_quantity{p}_{o}_{r}')
                    s.O_start[p][o][r] = model.NewIntVar(0, i.M, f'O_start{p}_{o}_{r}')
                    s.O_setup[p][o][r] = model.NewBoolVar(f'O_setup{p}_{o}_{r}')
                    s.O_end[p][o][r] = model.NewIntVar(0, i.M, f'O_end{p}_{o}_{r}')
                    s.O_executed[p][o][r] = model.NewBoolVar(f'O_executed{p}_{o}_{r}')
                    s.D_setup[p][o][r] = [model.NewBoolVar(f'D_setup{p}_{o}_{r}_{c}') for c in range(i.nb_settings)]
        for p2 in range(nb_projects):
            nb_ops2 = i.O_size[p2]
            s.precedes[p][p2] = init_3D(nb_ops, nb_ops2, i.nb_resources, None)
            for o in range(nb_ops):
                for o2 in range(nb_ops2):
                    for r in range(i.nb_resources):
                        if i.require(p, o, r) and i.require(p2, o2, r):
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

# Cmax computation
def c1(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        model.Add(s.Cmax >= s.E_end[p][i.project_head(p)])
    return model, s

# End of non-outsourced item
def c2(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            for o in i.last_operations(p, e):
                for r in i.required_resources(p, o):
                    model.Add(s.E_end[p][e] - i.M*s.O_executed[p][o][r] - i.real_time_scale(p,o)*s.O_end[p][o][r] >= -1 * i.M)
    return model, s

# End of outsourced item
def c3(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            model.Add(s.E_end[p][e] - s.E_prod_start[p][e] - i.M*s.E_outsourced[p][e] >= (i.outsourcing_time[p][e] - i.M))
    return model, s

# Physical start of non-outsourced item
def c4(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                if not i.is_design[p][o]:
                    for r in i.required_resources(p, o):
                        model.Add(s.E_prod_start[p][e] + i.M*s.O_executed[p][o][r] - i.real_time_scale(p,o)*s.O_start[p][o][r] <= i.M)
    return model, s

# Subcontract only if possible (known supplier)
def c5(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            if not i.external[p][e]:
                model.Add(s.E_outsourced[p][e] <= 0)
    return model, s

# Subcontract a whole branch of elements
def c6(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            for e2 in i.loop_items(p):
                if e2 != e and i.direct_assembly[p][e][e2]:
                    model.Add(s.E_outsourced[p][e2] - s.E_outsourced[p][e] >= 0)
    return model, s

# Available quantity of raw material before purchase
def c7(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if not i.finite_capacity[r]:
            terms = []
            for p in i.loop_projects():
                for o in i.loop_operations(p):
                    if i.require(p, o, r):
                        terms.append(i.quantity_needed[r][p][o]*s.O_uses_init_quantity[p][o][r])
            if len(terms)>0:
                model.Add(sum(terms) <= i.init_quantity[r])
    return model, s

# Is an operation executed with the init quantity (before purchase)?
def c8(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r in i.required_resources(p, o):
                if not i.finite_capacity[r]:
                    model.Add(i.M*s.O_uses_init_quantity[p][o][r] - s.O_executed[p][o][r] + i.real_time_scale(p,o)*s.O_start[p][o][r] >=  i.purchase_time[r] - 1)
    return model, s

# Complete execution of an operation on all required types of resources
def c9(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                for rt in range(i.nb_resource_types):
                    if i.resource_type_needed[p][o][rt]:
                        terms = []
                        for r in i.resources_by_type(rt):
                            terms.append(s.O_executed[p][o][r])
                        model.Add(s.E_outsourced[p][e] + sum(terms) == 1)
    return model, s

# Simultaneous operations (sync of resources mandatory)
def c10(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            if i.simultaneous[p][o]:
                for r in i.required_resources(p, o):
                    for v in i.required_resources(p, o):
                        if r != v:
                            model.Add(s.O_start[p][o][r] - s.O_start[p][o][v] + i.M*s.O_executed[p][o][r] + i.M*s.O_executed[p][o][v] <= 2*i.M)
    return model, s

# End of an operation according to the execution time and start time
def c11(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r in i.required_resources(p, o):
                if i.finite_capacity[r]:
                    model.Add(i.real_time_scale(p,o)*s.O_end[p][o][r] - i.real_time_scale(p,o)*s.O_start[p][o][r] - i.M*s.O_executed[p][o][r] >= i.execution_time[r][p][o] - i.M)
    return model, s

# Precedence relations between operations of one element
def c12(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o1 in range(start, end):
                for o2 in range(start, end):
                    if o1 != o2 and i.precedence[p][e][o1][o2]:
                        for r in i.required_resources(p, o1):
                            for v in i.required_resources(p, o2):
                                model.Add(i.real_time_scale(p,o1)*s.O_start[p][o1][r] - i.real_time_scale(p,o2)*s.O_end[p][o2][v] - i.M*s.O_executed[p][o1][r] - i.M*s.O_executed[p][o2][v] >= -2*i.M)
    return model, s

# Start time of parent' production
def c13(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.loop_items(p):
                if e1 != e2 and i.direct_assembly[p][e1][e2]:
                    model.Add(s.E_prod_start[p][e1] - s.E_end[p][e2] >= 0)
    return model, s

# Start time after design validation
def c14(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            model.Add(s.E_prod_start[p][e] - s.E_validated[p][e] >= 0)
    return model, s

# Design validation only after parent' validation
def c15(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.loop_items(p):
                if e1 != e2 and i.direct_assembly[p][e1][e2]:
                    model.Add(s.E_validated[p][e2] - s.E_validated[p][e1] >= 0)
    return model, s

# Start of any operation only after parent' validation
def c16(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e1 in i.loop_items(p):
            for e2 in i.loop_items(p):
                if e1 != e2 and i.direct_assembly[p][e1][e2]:
                    start, end = i.get_operations_idx(p, e2)
                    for o in range(start, end):
                        for r in i.required_resources(p, o):
                            model.Add(i.real_time_scale(p,o)*s.O_start[p][o][r] - s.E_validated[p][e1] -i.M*s.O_executed[p][o][r] >= -1 * i.M)
    return model, s

# No more than one direct predecessor (by resource)
def c17(model: cp_model.CpModel, i: Instance, s: Solution):
    for p1 in i.loop_projects():
        for o1 in i.loop_operations(p1):
            for r in i.required_resources(p1,o1):
                if i.finite_capacity[r]:
                    terms = []
                    for p2 in i.loop_projects():
                        for o2 in i.loop_operations(p2):
                            if not i.is_same(p1,p2,o1,o2) and i.require(p2,o2,r):
                                terms.append(s.precedes[p1][p2][o1][o2][r])
                    if len(terms)>0:
                        model.Add(sum(terms) <= 1)
    return model, s

# No more than one direct successor (by resource)
def c18(model: cp_model.CpModel, i: Instance, s: Solution):
    for p1 in i.loop_projects():
        for o1 in i.loop_operations(p1):
            for r in i.required_resources(p1,o1):
                if i.finite_capacity[r]:
                    terms = []
                    for p2 in i.loop_projects():
                        for o2 in i.loop_operations(p2):
                            if i.require(p2,o2,r):
                                terms.append(s.precedes[p2][p1][o2][o1][r])
                    if len(terms)>0:
                        model.Add(sum(terms) <= 1)
    return model, s

# No operation can be its own successor or predecessor
def c19(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            terms = []
            for p in i.loop_projects():
                for o in i.loop_operations(p): 
                    if i.require(p,o,r):
                        terms.append(s.precedes[p][p][o][o][r])
            if len(terms)>0:
                model.Add(sum(terms) == 0)
    return model, s

# Total number of operations in a resource (capacity)
def c20(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            terms_pos = []
            terms_neg = []
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    terms_pos.append(s.precedes[p1][p2][o1][o2][r])
            for p in i.loop_projects():
                for o in i.loop_operations(p): 
                    if i.require(p,o,r):
                        terms_neg.append(s.O_executed[p][o][r])
            if len(terms_pos)>0 or len(terms_neg)>0:
                model.Add(sum(terms_pos) - sum(terms_neg)>= -1)
    return model, s

# Precedence only for operations executed by the resource
def c21(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    model.Add(s.O_executed[p1][o1][r] - s.precedes[p1][p2][o1][o2][r] >= 0)                     
    return model, s

# Precedence only for operations executed by the resource (other way)
def c22(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    model.Add(s.O_executed[p2][o2][r] - s.precedes[p1][p2][o1][o2][r] >= 0) 
    return model, s

# Start operation only after the end of its predecessor (by resource)
def c23(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    terms = []
                                    for c in range(i.nb_settings):
                                        terms.append(i.design_setup[r][c] * s.D_setup[p1][o1][r][c])
                                    model.Add(i.real_time_scale(p1,o1)*s.O_start[p1][o1][r] -sum(terms) - i.real_time_scale(p2,o2) * s.O_end[p2][o2][r] - i.operation_setup[r]*s.O_setup[p1][o1][r] - i.M*s.precedes[p1][p2][o1][o2][r]>= -1 * i.M)
    return model, s

# Operation setups
def c24(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    weight = 0
                                    for ot in range(i.nb_ops_types):
                                        weight += 1 if i.operation_family[p1][o1][ot] != i.operation_family[p2][o2][ot] else 0
                                    model.Add(weight*s.precedes[p1][p2][o1][o2][r] - 2*s.O_setup[p1][o1][r] <= 0)
    return model, s

# Setups for design parameters
def c25(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if i.finite_capacity[r]:
            for p1 in i.loop_projects():
                for o1 in i.loop_operations(p1): 
                    if i.require(p1,o1,r):
                        for p2 in i.loop_projects():
                            for o2 in i.loop_operations(p2): 
                                if i.require(p2,o2,r) and not i.is_same(p1,p2,o1,o2):
                                    for c in range(i.nb_settings):
                                        model.Add(abs(i.design_value[p1][o1][c]-i.design_value[p2][o2][c])*s.precedes[p1][p2][o1][o2][r] - (i.design_value[p1][o1][c]+i.design_value[p2][o2][c])*s.D_setup[p1][o1][r][c] <= 0)
    return model, s

# Validation of an element
def c26(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            for o in range(start, end):
                if i.is_design[p][o]:
                    for r in i.required_resources(p,o):
                        model.Add(s.E_validated[p][e] - i.M*s.O_executed[p][o][r] - i.real_time_scale(p,o)*s.O_end[p][o][r] >= -1 * i.M)
    return model, s

# No double execution on the same type of resources (by operation)
def c27(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for r1 in i.required_resources(p,o):
                for r2 in i.required_resources(p,o):
                    if r1 != r2 and i.get_resource_familly(r1) == i.get_resource_familly(r2):
                        model.Add(s.O_executed[p][o][r1] + s.O_executed[p][o][r2] <= 1)
    return model, s

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