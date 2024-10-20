import pandas as pd
import argparse
import glob
import os
from common import directory
import pickle
from model.agent import MAPPO_Losses
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# =*= EXPERIMENTS: RESULTS ANALYSIS =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

problem_sizes = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
solution_types = ['exact', 'gns']
index_column = 'index'

# Combine the solutions of all instances by size and by type of solver
def combine_results_by_size_and_type(path:str, type: str):
    csv_files = glob.glob(os.path.join(path, 'solution_'+type+'_*.csv'))
    if csv_files:
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        return(pd.concat(dfs, ignore_index=True))
    else:
        print(f"No files at path={path} for solution type={type}")
        return None

# Combine all solution in a single object
def combine_all_results(basic_path: str):
    combined_solutions = {}
    for size in problem_sizes:
        combined_solutions[size] = {}
        path = basic_path+size+'/'
        for type in solution_types:
            combined = combine_results_by_size_and_type(path=path, type=type)
            if combined is not None:
                combined_solutions[size][type] = combined.sort_values(by=index_column)
                print(combined_solutions[size][type])
    print(combined_solutions)

def display_one(losses: list[Tensor], loss_name: str):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=loss_name)
    plt.xlabel('MAPPO Iteration')
    plt.ylabel('Loss')
    plt.title('Training Convergence: '+loss_name)
    plt.legend()
    plt.grid(True)
    plt.show()

# Display losses
def display_losses(path: str):
    with open(path, 'rb') as file:
        losses: MAPPO_Losses = pickle.load(file)
        display_one(losses.value_loss, 'Shared value loss')
        for agent in losses.agents:
            display_one(agent.policy_loss, 'Agent: '+agent.name+' - Policy loss')
            display_one(agent.entropy_bonus, 'Agent: '+agent.name+' - Entropy bonus')

'''
    TEST WITH
    python results_analysis.py --path=./
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 results analysis")
    parser.add_argument("--path", help="Path of the test instances", required=True)
    args = parser.parse_args()
    instances_path = args.path+directory.instances+'/test/'
    losses_path = args.path+directory.models+'/validation.pkl'
    combine_all_results(basic_path=instances_path)
    display_losses(losses_path)