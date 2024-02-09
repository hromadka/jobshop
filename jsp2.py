# from Google OR-Tools 
# https://developers.google.com/optimization/scheduling/job_shop


import collections
from ortools.sat.python import cp_model

jobs_data_simple = [  # task = (machine_id, processing_time).
    [(0, 3), (1, 2), (2, 2)],  # Job0
    [(0, 2), (2, 1), (1, 4)],  # Job1
    [(1, 4), (2, 3)],  # Job2
]

# Muth & Thompson 6x6 benchmark
jobs_data_6x6 = [
    [(3,1), (1,3), (2,6), (4,7), (6,3), (5,6)],
    [(2,8), (3,5), (5,10), (6,10), (1,10), (4,4)],
    [(3,5), (4,4), (6,8), (1,9), (2,1), (5,7)],
    [(2,5), (1,5), (3,5), (4,3), (5,8), (6,9)],
    [(3,9), (2,3), (5,5), (6,4), (1,3), (4,1)],
    [(2,3), (4,3), (6,9), (1,10), (5,4), (3,1)]
]

jobs_data_10x10 = [
    [(1,29), (1,43), (2,91), (2,81), (3,14), (3,84), (2,46), (3,31), (1,76), (2,85)],
    [(2,78), (3,90), (1,85), (3,95), (1,6), (2,2), (1,37), (1,86), (2,69), (1,13)],
    [(3,9), (5,75), (4,39), (1,71), (2,22), (6,52), (4,61), (2,46), (4,76), (3,61)],
    [(4,36), (10,11), (3,74), (5,99), (6,61), (4,95), (3,13), (6,74), (6,51), (7,7)],
    [(5,49), (4,69), (9,90), (7,9), (4,26), (9,48), (7,32), (5,32), (3,85), (9,64)],
    [(6,11), (2,28), (6,10), (9,52), (5,69), (10,72), (6,21), (7,88), (10,11), (10,76)],
    [(7,62), (7,46), (8,12), (8,85), (9,21), (1,47), (10,32), (9,19), (7,40), (6,47)],
    [(8,56), (6,46), (7,89), (4,98), (8,49), (7,65), (9,89), (10,48), (8,89), (4,52)],
    [(9,44), (8,72), (10,45), (10,22), (10,72), (5,6), (8,30), (8,36), (5,26), (5,90)],
    [(10,21), (9,30), (5,33), (6,43), (7,53), (8,25), (5,55), (4,79), (9,74), (8,45)]
]


jobs_data = jobs_data_10x10

###################################################################################


machines_count = 1 + max(task[0] for job in jobs_data for task in job)
all_machines = range(machines_count)
# Computes horizon dynamically as the sum of all durations.
horizon = sum(task[1] for job in jobs_data for task in job)

model = cp_model.CpModel()

# Named tuple to store information about created variables.
task_type = collections.namedtuple("task_type", "start end interval")
# Named tuple to manipulate solution information.
assigned_task_type = collections.namedtuple(
    "assigned_task_type", "start job index duration"
)

# Creates job intervals and add to the corresponding machine lists.
all_tasks = {}
machine_to_intervals = collections.defaultdict(list)

for job_id, job in enumerate(jobs_data):
    for task_id, task in enumerate(job):
        machine, duration = task
        suffix = f"_{job_id}_{task_id}"
        start_var = model.NewIntVar(0, horizon, "start" + suffix)
        end_var = model.NewIntVar(0, horizon, "end" + suffix)
        interval_var = model.NewIntervalVar(
            start_var, duration, end_var, "interval" + suffix
        )
        all_tasks[job_id, task_id] = task_type(
            start=start_var, end=end_var, interval=interval_var
        )
        machine_to_intervals[machine].append(interval_var)

# Create and add disjunctive constraints.
for machine in all_machines:
    model.AddNoOverlap(machine_to_intervals[machine])

# Precedences inside a job.
for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):
        model.Add(
            all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
        )

# Makespan objective.
obj_var = model.NewIntVar(0, horizon, "makespan")
model.AddMaxEquality(
    obj_var,
    [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
)
model.Minimize(obj_var)

# Call the solver.
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Display results.
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution:")
    # Create one list of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            assigned_jobs[machine].append(
                assigned_task_type(
                    start=solver.Value(all_tasks[job_id, task_id].start),
                    job=job_id,
                    index=task_id,
                    duration=task[1],
                )
            )

    # Create per machine output lines.
    output = ""
    for machine in all_machines:
        # Sort by starting time.
        assigned_jobs[machine].sort()
        sol_line_tasks = "Machine " + str(machine) + ": "
        sol_line = "           "

        for assigned_task in assigned_jobs[machine]:
            name = f"job_{assigned_task.job}_task_{assigned_task.index}"
            # Add spaces to output to align columns.
            sol_line_tasks += f"{name:15}"

            start = assigned_task.start
            duration = assigned_task.duration
            sol_tmp = f"[{start},{start + duration}]"
            # Add spaces to output to align columns.
            sol_line += f"{sol_tmp:15}"

        sol_line += "\n"
        sol_line_tasks += "\n"
        output += sol_line_tasks
        output += sol_line

    # Finally print the solution found.
    print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
    print(output)
else:
    print("No solution found.")

# Statistics.
print("\nStatistics")
print(f"  - conflicts: {solver.NumConflicts()}")
print(f"  - branches : {solver.NumBranches()}")
print(f"  - wall time: {solver.WallTime()}s")


print("done")