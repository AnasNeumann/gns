import pickle
import os
from common import set_memory_limit, load_instance
from model import Instance
import argparse
import pandas as pd
from ortools.sat.python import cp_model
import time as systime

MAX_COMPUTING_HOURS = 3
MAX_RAM = 13

def init_vars(model: cp_model.CpModel):
    return model

def init_objective_function(model: cp_model.CpModel):
    obj_var = None
    return model, obj_var

def c1(model: cp_model.CpModel):
    return model

def c2(model: cp_model.CpModel):
    return model

def c3(model: cp_model.CpModel):
    return model

def c4(model: cp_model.CpModel):
    return model

def c5(model: cp_model.CpModel):
    return model

def c6(model: cp_model.CpModel):
    return model

def c7(model: cp_model.CpModel):
    return model

def c8(model: cp_model.CpModel):
    return model

def c9(model: cp_model.CpModel):
    return model

def c10(model: cp_model.CpModel):
    return model

def c11(model: cp_model.CpModel):
    return model

def c12(model: cp_model.CpModel):
    return model

def c13(model: cp_model.CpModel):
    return model

def c14(model: cp_model.CpModel):
    return model

def c15(model: cp_model.CpModel):
    return model

def c16(model: cp_model.CpModel):
    return model

def c17(model: cp_model.CpModel):
    return model

def c18(model: cp_model.CpModel):
    return model

def c19(model: cp_model.CpModel):
    return model

def c20(model: cp_model.CpModel):
    return model

def c21(model: cp_model.CpModel):
    return model

def c22(model: cp_model.CpModel):
    return model

def c23(model: cp_model.CpModel):
    return model

def c24(model: cp_model.CpModel):
    return model

def c25(model: cp_model.CpModel):
    return model

def c26(model: cp_model.CpModel):
    return model

def c27(model: cp_model.CpModel):
    return model

def solve_one(instance: Instance, solution_path):
    start_time = systime.time()
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_COMPUTING_HOURS * 60.0 
    model, obj_var = init_objective_function(init_vars(model))
    for constraint in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27]:
        model = constraint(model)
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