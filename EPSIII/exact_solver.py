import pickle
import os
from common import set_memory_limit, load_instance
from model import Instance
import argparse
import pandas as pd
from ortools.sat.python import cp_model

def solve_one(instance: Instance, solution_path):
    solution = None

    # TODO
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3 * 60.0 

    solutions_df = pd.DataFrame({'index': instance.id, 'value': solution, 'status': None, 'gap': None})
    solutions_df.to_csv(solution_path, index=False)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=True)
    parser.add_argument("--number", help="Number of the solved instance", required=True)
    args = parser.parse_args()
    INSTANCE_PATH = './instances/test/'+args.size+'/instance_'+args.number+'.pkl'
    SOLUTION_PATH = './instances/test/'+args.size+'/solution_'+args.number+'.csv'
    set_memory_limit(13)
    instance = load_instance(INSTANCE_PATH)
    solve_one(instance, SOLUTION_PATH)