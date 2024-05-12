from ortools.sat.python import cp_model
import sys
import pandas as pd
from common import load_instances

# CONFIGURATION AND LOAD INSTANCES
INSTANCES_TYPES = sys.argv[1] # train or test
INSTANCES_PATH = './FJS/instances/'+INSTANCES_TYPES
END_DATE = 1
START_DATE = 0
CHOSEN_RESOURCE = 3
instances = load_instances(INSTANCES_PATH)

# SOLVE ALL INSTANCES ONE BY ONE
solutions = []
for i in instances:
    print("============================")
    print("=*= START A NEW INSTANCE =*=")
    print("============================")
    
    # Load data and display the instance
    print(i)
    jobs = i['jobs']
    resources_types = i['resources']
    num_jobs = len(jobs)
    num_resource_types = len(resources_types)
    all_operations = {}
    resources_by_type = {type_id: [] for type_id in range(num_resource_types)}
    
    resources_intervals = {type_id: [[] for _ in range(resources_types[type_id])] for type_id in range(num_resource_types)}

    # Flatten the resources
    for resource_type, count in enumerate(resources_types):
        for resource_id in range(count):
            resources_by_type[resource_type].append(resource_id)

    # CREATE THE WHOLE MODEL
    model = cp_model.CpModel()
    for job_id, job in enumerate(jobs):
        for operation_id, (resource_type, duration) in enumerate(job):
            
            # OPERATIONS VARIABLES: "start", "end", and "resource selection (select and resource)" variables for each operation
            suffix = f'_{job_id}_{operation_id}'
            start_var = model.NewIntVar(0, sum(d for j in jobs for _, d in j), f'start{suffix}')
            end_var = model.NewIntVar(0, sum(d for j in jobs for _, d in j), f'end{suffix}')
            chosen_resource = model.NewIntVar(0, len(resources_by_type[resource_type]) - 1, f'resource{suffix}')
            resource_selection = [model.NewBoolVar(f'select{suffix}_r{resource_id}') for resource_id in resources_by_type[resource_type]]

            intervals_for_resources = []
            for resource_id in resources_by_type[resource_type]:
                is_used = resource_selection[resource_id]
                # INTERVAL VARIABLE: "interval" for each resource that could be chosen (optional is "is_used")!
                interval_var = model.NewOptionalIntervalVar(start_var, duration, end_var, is_used, f'interval{suffix}_r{resource_id}')
                intervals_for_resources.append(interval_var)
                # FIRST CONSTRAINT: the boolean selection variable and the integer one (num of the type of resource) must be consistent
                model.Add(chosen_resource == resource_id).OnlyEnforceIf(is_used)
            resources_intervals[resource_type][resource_id].append(interval_var)

            # SECOND CONSTRAINT: exactly one and only one resource selected by operation
            model.Add(sum(resource_selection) == 1)

            # Store all varaibles of an operation into "all_operations"
            all_operations[(job_id, operation_id)] = (start_var, end_var, intervals_for_resources, chosen_resource)

    # THIRD CONSTRAINT: no overlapp between intervals for each resources
    for intervals_by_type in resources_intervals.values():
        for intervals_by_resource in intervals_by_type:
            model.AddNoOverlap(intervals_by_resource)

    # FOURTH CONSTRAINT: precedence relationship (end before start)
    for job_id, job in enumerate(jobs):
        for operation_id in range(len(job) - 1):
            model.Add(all_operations[(job_id, operation_id)][END_DATE] <= all_operations[(job_id, operation_id + 1)][START_DATE])

    # OBJECTIVE: minimize the makespan (as a new variable and fifth constraint)
    obj_var = model.NewIntVar(0, sum(d for j in jobs for _, d in j), 'makespan')
    model.AddMaxEquality(obj_var, [end_var for _, end_var, _, _ in all_operations.values()])
    model.Minimize(obj_var)

    # SOLVE: solve the model and display the results
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Minimum makespan: {solver.Value(obj_var)}')
        solutions.append(solver.Value(obj_var))
        for job_id, job in enumerate(jobs):
            print(f'Job {job_id}:')
            for operation_id, (resource_type, duration) in enumerate(job):
                start = solver.Value(all_operations[(job_id, operation_id)][START_DATE])
                end = solver.Value(all_operations[(job_id, operation_id)][END_DATE])
                chosen_resource = solver.Value(all_operations[(job_id, operation_id)][CHOSEN_RESOURCE])
                print(f'  Task {operation_id} (Resource Type {resource_type}, Chosen Resource {chosen_resource}, Duration {duration}): Start={start}, End={end}')
    else:
        print('No solution found.')

# SAVE : the solution as a dataFrame in CSV
solutions_df = pd.DataFrame({
    'index': range(0, len(solutions)),
    'values': solutions
})
solutions_df.to_csv(INSTANCES_PATH+'/optimal.csv', index=False)
print("====================")
print("=*= END OF FILE! =*=")
print("====================")
"""=================================================================================================================================================
                                                -=*=- END OF FILE -=*=-
=================================================================================================================================================="""