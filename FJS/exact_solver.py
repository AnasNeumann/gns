from ortools.sat.python import cp_model

# Define the data
num_jobs = 3
num_machines = 3

jobs_data = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0
    [(0, 2), (2, 1), (1, 4)],  # Job 1
    [(1, 4), (2, 3)]           # Job 2
]

# Create the model
model = cp_model.CpModel()

# Create variables
all_tasks = {}
machine_to_intervals = {machine: [] for machine in range(num_machines)}

# Create a task (interval) for each operation of each job
for job_id, job in enumerate(jobs_data):
    for task_id, (machine, duration) in enumerate(job):
        suffix = f'_{job_id}_{task_id}'
        start_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), f'start{suffix}')
        end_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), f'end{suffix}')
        interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval{suffix}')
        all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var)
        machine_to_intervals[machine].append(interval_var)

# Add constraints for no overlapping intervals on the same machine
for machine, intervals in machine_to_intervals.items():
    model.AddNoOverlap(intervals)

# Add precedence constraints within each job
for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):
        model.Add(all_tasks[(job_id, task_id)][1] <= all_tasks[(job_id, task_id + 1)][0])

# Objective: minimize the makespan
obj_var = model.NewIntVar(0, sum(d for j in jobs_data for _, d in j), 'makespan')
model.AddMaxEquality(obj_var, [end_var for _, end_var, _ in all_tasks.values()])
model.Minimize(obj_var)

# Create the solver and solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Display the results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Minimum makespan: {solver.Value(obj_var)}')
    for job_id, job in enumerate(jobs_data):
        print(f'Job {job_id}:')
        for task_id, (machine, duration) in enumerate(job):
            start = solver.Value(all_tasks[(job_id, task_id)][0])
            end = solver.Value(all_tasks[(job_id, task_id)][1])
            print(f'  Task {task_id} (Machine {machine}, Duration {duration}): Start={start}, End={end}')
else:
    print('No solution found.')
