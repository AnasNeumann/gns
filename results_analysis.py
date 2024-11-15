import pandas as pd
import argparse
import glob
import os
from common import directory
import pickle
from model.agent import MAPPO_Losses
from torch import Tensor
import matplotlib.pyplot as plt

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

def display_one(losses: list[Tensor], loss_name: str, path: str):
    plt.figure(figsize=(10, 5))
    x_values = range(1, len(losses) + 1)
    plt.plot(x_values, losses, label=loss_name)
    plt.xlabel('MAPPO Iteration')
    plt.ylabel('Loss')
    plt.title('Training Convergence: '+loss_name)
    plt.legend()
    plt.grid(True)
    for x in range(8, len(losses) + 1, 8):
        plt.axvline(x=x, color='black', linestyle='--', linewidth=1)
    plt.savefig(path+"/"+loss_name+".png")
    plt.show()

# Display losses
def display_losses(model_path: str, result_path: str, last: int):
    value_losses: list[float] = []
    outsourcing_losses: list[float] = []
    scheduling_losses: list[float] = []
    material_losses: list[float] = []
    for model_id in range(1, last + 1):
        with open(model_path+'/validation_'+str(model_id)+'.pkl', 'rb') as file:
            l: MAPPO_Losses = pickle.load(file)
            value_losses.extend(l.value_loss)
            for agent in l.agents:
                if agent.name == "outsourcing":
                    outsourcing_losses.extend(agent.policy_loss)
                elif agent.name == "scheduling":
                    scheduling_losses.extend(agent.policy_loss)
                else:
                    material_losses.extend(agent.policy_loss)
    display_one(value_losses, 'Shared value loss', result_path)
    display_one(outsourcing_losses, 'Outsoucing policy loss', result_path)
    display_one(scheduling_losses, 'Scheduling policy loss', result_path)
    display_one(material_losses, 'Material use policy loss', result_path)
    
'''
    TEST WITH
    python results_analysis.py --path=./ --last=7
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 results analysis")
    parser.add_argument("--path", help="Path of the test instances", required=True)
    parser.add_argument("--last", help="Last trained version of GNN model", required=True)
    args = parser.parse_args()
    instances_path = args.path+directory.instances+'/test/'
    model_path = args.path+directory.models
    result_path = args.path+directory.results
    #combine_all_results(basic_path=instances_path)
    display_losses(model_path=model_path, result_path=result_path, last=int(args.last))