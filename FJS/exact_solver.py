from ortools.sat.python import cp_model
import sys
import os
import pickle

# CONFIGURATION
INSTANCES_TYPES = sys.argv[1] # train or test
INSTANCES_PATH = './FJS/instances/'+INSTANCES_TYPES

# LOAD INSTANCES
instances = []
for i in os.listdir(INSTANCES_PATH):
    if i.endswith('.pkl'):
        file_path = os.path.join(INSTANCES_PATH, i)
        print(f"Loading data from: {i}...")
        with open(file_path, 'rb') as file:
            instances.append(pickle.load(file))

# SOLVE ALL INSTANCES ONE BY ONE
for i in instances:
    print("============================")
    print("=*= START A NEW INSTANCE =*=")
    print("============================")
    
    # Display the instance
    print(i)

    # Create variables
    model = cp_model.CpModel()
    num_jobs = len(i['jobs'])
    num_resource_types = len(i['resources'])
    jobs_data = i['jobs']
    resources = i['resources']
    all_tasks = {}
    type_to_intervals = {type_id: [] for type_id in range(num_resource_types)}
    type_to_resources = {type_id: [] for type_id in range(num_resource_types)}

    # Create variables for resources of each type
    for type_id, count in enumerate(resources):
        for resource_id in range(count):
            type_to_resources[type_id].append(resource_id)

    # Create a task (interval) for each operation of each job
    for job_id, job in enumerate(jobs_data):
        for task_id, (resource_type, duration) in enumerate(job):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), f'start{suffix}')
            end_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), f'end{suffix}')
            chosen_resource = model.NewIntVar(0, len(type_to_resources[resource_type]) - 1, f'resource{suffix}')
            interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval{suffix}')

            all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var, chosen_resource)

            # Add to the appropriate resource type's intervals
            for resource_id in type_to_resources[resource_type]:
                # Only consider this resource if it's chosen by this operation
                model.Add(chosen_resource == resource_id).OnlyEnforceIf(interval_var)
                type_to_intervals[resource_type].append(interval_var)

    # Add constraints for no overlapping intervals on the same resource type
    for resource_type, intervals in type_to_intervals.items():
        model.AddNoOverlap(intervals)

    # Add precedence constraints within each job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[(job_id, task_id)][1] <= all_tasks[(job_id, task_id + 1)][0])

    # Objective: minimize the makespan
    obj_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), 'makespan')
    model.AddMaxEquality(obj_var, [end_var for _, end_var, _, _ in all_tasks.values()])
    model.Minimize(obj_var)

    # Create the solver and solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Display the results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Minimum makespan: {solver.Value(obj_var)}')
        for job_id, job in enumerate(jobs_data):
            print(f'Job {job_id}:')
            for task_id, (resource_type, duration) in enumerate(job):
                start = solver.Value(all_tasks[(job_id, task_id)][0])
                end = solver.Value(all_tasks[(job_id, task_id)][1])
                chosen_resource = solver.Value(all_tasks[(job_id, task_id)][3])
                print(f'  Task {task_id} (Resource Type {resource_type}, Chosen Resource {chosen_resource}, Duration {duration}): Start={start}, End={end}')
    else:
        print('No solution found.')



print("====================")
print("=*= END OF FILE! =*=")
print("====================")
"""=================================================================================================================================================
                                                -=*=- END OF FILE -=*=-
=================================================================================================================================================="""