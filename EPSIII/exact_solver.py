import pickle
import os
from common import set_memory_limit, load_instance, init_several_1D, init_2D, init_several_2D, init_3D
from model import Instance, Solution, get_nb_projects, require, project_head, last_operations, required_resources, get_operations_idx, is_same, get_resource_familly, resources_by_type, real_time_scale, resources_by_type
import argparse
import pandas as pd
from ortools.sat.python import cp_model
import time as systime

MAX_COMPUTING_HOURS = 3
MAX_RAM = 13

def init_vars(model: cp_model.CpModel, i: Instance, s: Solution):
    s = Solution()
    nb_projects = get_nb_projects(i)
    elts_per_project = i.E_size[0]
    s.E_start = [[model.New(0, i.M, f'E_start_{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_outsourced = [[model.NewBoolVar(f'E_outsource{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_prod_start = [[model.NewIntVar(0, i.M, f'E_prod_start{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_end = [[model.NewIntVar(0, i.M, f'E_end{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.E_validated = [[model.NewIntVar(0, i.M, f'E_validated{p}_{e}') for e in range(elts_per_project)] for p in range(nb_projects)]
    s.O_uses_init_quantity, s.O_start, s.O_setup, s.O_end, s.O_executed, s.D_setup = init_several_1D(nb_projects, [None], 6)
    s.precedes = init_2D(nb_projects, nb_projects, [None])
    s.Cmax = model.NewIntVar(0, i.M, 'Cmax')
    for p in range(nb_projects):
        nb_ops = i.O_size[p]
        s.O_uses_init_quantity[p], s.O_start[p], s.O_setup[p], s.O_end[p], s.O_executed[p] = init_several_2D(nb_ops, i.nb_resources, [None], 5)
        s.D_setup[p] = init_3D(nb_ops, i.nb_resources, i.nb_settings, [None])
        for o in range(nb_ops):
            for r in range(i.nb_resources):
                if require(i, p, o, r):
                    s.O_uses_init_quantity[p][o][r] = model.NewBoolVar(f'O_uses_init_quantity{p}_{o}_{r}')
                    s.O_start[p][o][r] = model.NewIntVar(0, i.M, f'O_start{p}_{o}_{r}')
                    s.O_setup[p][o][r] = model.NewBoolVar(f'O_setup{p}_{o}_{r}')
                    s.O_end[p][o][r] = model.NewIntVar(0, i.M, f'O_end{p}_{o}_{r}')
                    s.O_executed[p][o][r] = model.NewBoolVar(f'O_executed{p}_{o}_{r}')
                    s.D_setup[p][o][r] = [model.NewBoolVar(f'D_setup{p}_{o}_{r}_{s}') for s in range(i.nb_settings)]
        for p2 in range(nb_projects):
            nb_ops2 = i.O_size[p2]
            s.precedes[p][p2] = init_3D(nb_ops, nb_ops2, i.nb_resources, [None])
            for o in range(nb_ops):
                for o2 in range(nb_ops2):
                    for r in range(i.nb_resources):
                        if require(i, p, o, r) and require(i, p2, o2, r):
                            s.precedes[p][p2][o][o2][r] = model.NewBoolVar(f'precedes{p}_{p2}_{o}_{o2}_{r}')
    return model, s

def init_objective_function(model: cp_model.CpModel, i: Instance, s: Solution):
    s.obj.append(s.Cmax * i.w_makespan)
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            if i.external[p][e]:
                s.obj.append(s.E_outsourced[p][e] * i.external_cost[p][e] * (1.0 - i.w_makespan))
    model.Minimize(sum(s.obj))
    return model, s

# Cmax computation
def c1(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        model.Add(s.Cmax >= s.E_end[p][project_head(i,p)])
    return model, s

# End of non-outsourced item
def c2(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            for o in last_operations(i, p, e):
                for r in required_resources(i, p, o):
                    model.Add(s.E_end[p][e] - i.M*s.O_executed[p][o][r] - real_time_scale(i,p,o)*s.O_end[p][o][r] >= -1.0*i.M)
    return model, s

# End of outsourced item
def c3(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            model.Add(s.E_end[p][e] - s.E_prod_start[p][e] - i.M*s.E_outsourced[p][e] >= (i.outsourcing_time[p][e] - i.M))
    return model, s

# Physical start of non-outsourced item
def c4(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            for o in get_operations_idx(i, p, e):
                if not i.is_design[p][o]:
                    for r in required_resources(i, p, o):
                        model.Add(s.E_prod_start[p][e] + i.M*s.O_executed[p][o][r] - real_time_scale(i,p,o)*s.O_start[p][o][r] <= i.M)
    return model, s

# Subcontract only if possible (known supplier)
def c5(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            if not i.external[p][e]:
                model.Add(s.E_outsourced[p][e] <= 0)
    return model, s

# Subcontract a whole branch of elements
def c6(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            for e2 in range(i.E_size[p]):
                if e2 != e and i.direct_assembly[p][e][e2]:
                    model.Add(s.E_outsourced[p][e2] - s.E_outsourced[p][e] >= 0)
    return model, s

# Available quantity of raw material before purchase
def c7(model: cp_model.CpModel, i: Instance, s: Solution):
    for r in range(i.nb_resources):
        if not i.finite_capacity[r]:
            constraint = cp_model.LinearExpr()
            for p in range(get_nb_projects(i)):
                for o in range(i.O_size[p]):
                    if require(i, p, o, r):
                        constraint += i.quantity_needed[r][p][o]*s.O_uses_init_quantity[p][o][r]
            model.Add(constraint <= i.init_quantity[r])
    return model, s

def reverse_scale(val, M):
    return (1.0 * val)/M

# Is an operation executed with the init quantity (before purchase)?
def c8(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for o in range(i.O_size[p]):
            for r in required_resources(p, o):
                if not i.finite_capacity[r]:
                    model.Add(s.O_uses_init_quantity[p][o][r] - s.O_executed[p][o][r] + reverse_scale(real_time_scale(i,p,o), i.M)*s.O_start[p][o][r] >= reverse_scale(i.purchase_time[r], i.M) - 1)
    return model, s

# Complete execution of an operation on all required types of resources
def c9(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            for o in get_operations_idx(i, p, e):
                for rt in i.nb_resource_types:
                    if i.resource_type_needed[p][o][rt]:
                        constraint = cp_model.LinearExpr()
                        constraint += s.E_outsourced[p][e]
                        for r in resources_by_type(i, rt):
                            constraint += s.O_executed[p][o][r]
                        model.Add(constraint == 1)
    return model, s

# Simultaneous operations (sync of resources mandatory)
def c10(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for o in range(i.O_size[p]):
            if i.simultaneous[p][o]:
                for r in required_resources(i, p, o):
                    for v in required_resources(i, p, o):
                        if r != v:
                            model.Add(s.O_start[p][o][r] - s.O_start[p][o][v] + i.M*s.O_executed[p][o][r] + i.M*s.O_executed[p][o][v] <= 2*i.M)
    return model, s

# End of an operation according to the execution time and start time
def c11(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for o in range(i.O_size[p]):
            for r in required_resources(i, p, o):
                if i.finite_capacity[r]:
                    model.Add(real_time_scale(i,p,o)*s.O_end[p][o][r] - real_time_scale(i,p,o)*s.O_start[p][o][r] - i.M*s.O_executed[p][o][r] >= i.execution_time[r][p][o] - i.M)
    return model, s

# Precedence relations between operations of one element
def c12(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            for o1 in get_operations_idx(i, p, e):
                for o2 in get_operations_idx(i, p, e):
                    if o1 != o2 and i.precedence[p][e][o1][o2]:
                        for r in required_resources(i, p, o1):
                            for v in required_resources(i, p, o2):
                                model.Add(real_time_scale(i,p,o1)*s.O_start[p][o1][r] - real_time_scale(i,p,o2)*s.O_end[p][o2][v] - i.M*s.O_executed[p][o1][r] - i.M*s.O_executed[p][o2][v] >= -2*i.M)
    return model, s

# Start time of parent' production
def c13(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e1 in range(i.E_size[p]):
            for e2 in range(i.E_size[p]):
                if e1 != e2 and i.direct_assembly[p][e1][e2]:
                    model.Add(s.E_prod_start[p][e1] - s.E_end[p][e2] >= 0)
    return model, s

# Start time after design validation
def c14(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e in range(i.E_size[p]):
            model.Add(s.E_prod_start[p][e] - s.E_validated[p][e] >= 0)
    return model, s

# Design validation only after parent' validation
def c15(model: cp_model.CpModel, i: Instance, s: Solution):
    for p in range(get_nb_projects(i)):
        for e1 in range(i.E_size[p]):
            for e2 in range(i.E_size[p]):
                if e1 != e2 and i.direct_assembly[p][e1][e2]:
                    model.Add(s.validated[p][e2] - s.E_validated[p][e1] >= 0)
    return model, s

# Start of any operation only after parent' validation
def c16(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# No more than one direct predecessor (by resource)
def c17(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# No more than one direct successor (by resource)
def c18(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# No operation can be its own successor or predecessor
def c19(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Total number of operations in a resource (capacity)
def c20(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Precedence only for operations executed by the resource
def c21(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Precedence only for operations executed by the resource (other way)
def c22(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Start operation only after the end of its predecessor (by resource)
def c23(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Operation setups
def c24(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Setups for design parameters
def c25(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# Validation of an element
def c26(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

# No double execution on the same type of resources (by operation)
def c27(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def solve_one(instance: Instance, solution_path):
    start_time = systime.time()
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_COMPUTING_HOURS * 60.0 
    model, solution = init_vars(model, instance)
    model, solution = init_objective_function(model, instance, solution)
    for constraint in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27]:
        model, solution = constraint(model, instance, solution)
    status = solver.Solve(model)
    computing_time = systime.time()-start_time
    if status == cp_model.OPTIMAL:
        solutions_df = pd.DataFrame({'index': instance.id, 'value': solver.Value(solver.ObjectiveValue()), 'status': 'optimal', 'computing_time': computing_time, 'max_time': MAX_COMPUTING_HOURS, 'max_memory': MAX_RAM})
    elif status == cp_model.FEASIBLE:
        solutions_df = pd.DataFrame({'index': instance.id, 'value': solver.Value(solver.ObjectiveValue()), 'status': 'feasible', 'computing_time': computing_time, 'max_time': MAX_COMPUTING_HOURS, 'max_memory': MAX_RAM})
    else:
        solutions_df = pd.DataFrame({'index': instance.id, 'value': -1, 'status': 'failure', 'computing_time': computing_time, 'max_time': MAX_COMPUTING_HOURS, 'max_memory': MAX_RAM})
    print(solutions_df)
    solutions_df.to_csv(solution_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=True)
    parser.add_argument("--number", help="Number of the solved instance", required=True)
    args = parser.parse_args()
    INSTANCE_PATH = './instances/test/'+args.size+'/instance_'+args.number+'.pkl'
    SOLUTION_PATH = './instances/test/'+args.size+'/solution_'+args.number+'.csv'
    set_memory_limit(MAX_RAM)
    instance = load_instance(INSTANCE_PATH)
    print("===* START SOLVING *===")
    solve_one(instance, SOLUTION_PATH)
    print("===* END OF FILE *===")