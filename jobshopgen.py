from job_shop_lib import JobShopInstance, Operation, ScheduledOperation, Schedule
from job_shop_lib.dispatching import Dispatcher
import numpy as np

CPU = 0
GPU = 1
DATA_CENTER = 2

job_1 = [Operation(CPU, 1), Operation(GPU, 1), Operation(DATA_CENTER, 7)]
job_2 = [Operation(GPU, 5), Operation(DATA_CENTER, 1), Operation(CPU, 1)]
job_3 = [Operation(DATA_CENTER, 1), Operation(CPU, 3), Operation(GPU, 2)]

jobs = [job_1, job_2, job_3]

instance = JobShopInstance(
    jobs,
    name="Example",
    # Any extra parameters are stored inside the
    # metadata attribute as a dictionary:
    lower_bound=7,
)

print("Number of jobs:", instance.num_jobs)
print("Number of machines:", instance.num_machines)
print("Number of operations:", instance.num_operations)
print("Name:", instance.name)
print("Is flexible?:", instance.is_flexible)
print("Max operation time:", instance.max_duration)
print("Machine loads:", instance.machine_loads)


arrinstances = np.array(instance.durations_matrix)
print(arrinstances)

arrmatrix = np.array(instance.machines_matrix)
print(arrmatrix)

first_operation = job_1[0]
print("Machine id:", first_operation.machine_id)
print("Duration:", first_operation.duration)
# If the operation only has one machine, we can use the `machine_id` property
# instead of the `machines` attribute:
print("Job id:", first_operation.job_id)
print("Position in the job:", first_operation.position_in_job)
print("Operation id:", first_operation.operation_id)
print("String representation:", str(first_operation))


cpu_operations = []
gpu_operations = []
data_center_operations = []

# Split the operations into three lists, one for each machine
for job in instance.jobs:
    for operation in job:
        if operation.machine_id == CPU:
            cpu_operations.append(operation)
        elif operation.machine_id == GPU:
            gpu_operations.append(operation)
        elif operation.machine_id == DATA_CENTER:
            data_center_operations.append(operation)

# Schedule the operations as they are ordered in the instance
def schedule_operations(operations, machine_id, start_time=0):
    machine_schedule = []
    for operation in operations:
        machine_schedule.append(
            ScheduledOperation(operation, start_time, machine_id)
        )
        start_time += 7

    return machine_schedule


cpu_schedule = schedule_operations(cpu_operations, CPU)
gpu_schedule = schedule_operations(gpu_operations, GPU)
data_center_schedule = schedule_operations(
    data_center_operations, DATA_CENTER
)

instance_schedule = [cpu_schedule, gpu_schedule, data_center_schedule]
schedule = Schedule(instance, instance_schedule)

print(f"Is the schedule complete? {schedule.is_complete()}")
print("Makespan:", schedule.makespan())

raw_solution = [cpu_operations, gpu_operations, data_center_operations]

print("optimizing...")
optimized_schedule = Dispatcher.create_schedule_from_raw_solution(
    instance, raw_solution
)
print(f"Is the schedule complete? {optimized_schedule.is_complete()}")
print("Makespan:", optimized_schedule.makespan())

print("done!")