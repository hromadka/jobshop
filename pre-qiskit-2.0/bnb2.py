class Node:
    def __init__(self, level, schedule, cost):
        self.level = level
        self.schedule = schedule
        self.cost = cost

def calculate_cost(schedule):
    max_end_time = max([max(task[1] for task in machine_schedule) for machine_schedule in schedule])
    return max_end_time

def branch_and_bound(job_list, machine_count):
    n = len(job_list)
    total_nodes = 0

    # Initialize the root node
    root = Node(level=0, schedule=[[] for _ in range(machine_count)], cost=0)
    priority_queue = [root]

    while priority_queue:
        node = priority_queue.pop(0)

        # If all jobs are assigned
        if node.level == n:
            return node.schedule

        # Go to the next level
        for machine in range(machine_count):
            new_schedule = [schedule[:] for schedule in node.schedule]
            new_schedule[machine].append(job_list[node.level])
            new_cost = calculate_cost(new_schedule)

            child = Node(level=node.level + 1, schedule=new_schedule, cost=new_cost)

            # Add child to the queue
            priority_queue.append(child)
            total_nodes += 1

        # Sort the queue based on the lower bound
        priority_queue.sort(key=lambda x: x.cost)

    return None

def print_schedule(schedule):
    for i, machine_schedule in enumerate(schedule):
        print(f"Machine {i+1}:")
        for task in machine_schedule:
            print(f"Job {task[0]} (Duration: {task[1]})")
        print()

# Example usage
job_list = [
    [(1, 3), (2, 2), (3, 1)],  # Job 1
    [(2, 2), (3, 1), (1, 3)],  # Job 2
    [(3, 1), (1, 3), (2, 2)]   # Job 3
]
machine_count = 3

schedule = branch_and_bound(job_list, machine_count)
if schedule:
    print("Optimal Job Shop Schedule:")
    print_schedule(schedule)
else:
    print("No feasible solution found.")
