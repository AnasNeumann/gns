import pickle
import os
from common import set_memory_limit, load_instance, init_several_1D, init_2D, init_several_2D, init_3D
from model import Instance, Solution, get_nb_projects, require
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
    obj_var = None
    return model, s, obj_var

def c1(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c2(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c3(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c4(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c5(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c6(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c7(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c8(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c9(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c10(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c11(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c12(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c13(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c14(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c15(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c16(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c17(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c18(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c19(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c20(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c21(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c22(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c23(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c24(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c25(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c26(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def c27(model: cp_model.CpModel, i: Instance, s: Solution):
    return model, s

def solve_one(instance: Instance, solution_path):
    start_time = systime.time()
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_COMPUTING_HOURS * 60.0 
    model, solution = init_vars(model, instance)
    model, solution, obj_var = init_objective_function(model, instance, solution)
    for constraint in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27]:
        model, solution = constraint(model, instance, solution)
    status = solver.Solve(model)
    computing_time = systime.time()-start_time
    if status == cp_model.OPTIMAL:
        solutions_df = pd.DataFrame({'index': instance.id, 'value': solver.Value(obj_var), 'status': 'optimal', 'computing_time': computing_time, 'max_time': MAX_COMPUTING_HOURS, 'max_memory': MAX_RAM})
    elif status == cp_model.FEASIBLE:
        solutions_df = pd.DataFrame({'index': instance.id, 'value': solver.Value(obj_var), 'status': 'feasible', 'computing_time': computing_time, 'max_time': MAX_COMPUTING_HOURS, 'max_memory': MAX_RAM})
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