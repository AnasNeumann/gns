import pandas as pd
import argparse
import glob
import os
from common import directory

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

'''
    TEST WITH
    python results_analysis.py --path=./
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 results analysis")
    parser.add_argument("--path", help="Path of the test instances", required=True)
    args = parser.parse_args()
    basic_path = args.path+directory.instances+'/test/'
    combine_all_results(basic_path=basic_path)