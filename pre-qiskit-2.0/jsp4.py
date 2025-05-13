# another ChatGPT 3.5 implementation - 3/8/2024
# prompt = please write an advanced python program to solve the job shop scheduling problem using branch and bound

import numpy as np
import heapq
import copy

class Node:
    def __init__(self, job=None, sequence=None, time=0, lower_bound=0):
        self.job = job
        self.sequence = sequence
        self.time = time
        self.lower_bound = lower_bound

def calculate_lower_bound(node, jobs, machines):
    remaining_jobs = jobs - set(node.sequence)
    lb = node.time
    for job in remaining_jobs:
        next_machine = node.sequence[-1] if node.sequence else -1
        lb += machines[next_machine + 1][job]
    return lb

def find_next_job_sequence(node, jobs, machines):
    min_time = float('inf')
    next_job = None
    for job in jobs - set(node.sequence):
        next_machine = node.sequence[-1] if node.sequence else -1
        time = node.time + machines[next_machine + 1][job]
        if time < min_time:
            min_time = time
            next_job = job
    return next_job

def branch_and_bound(jobs, machines):
    heap = []
    root = Node()
    root.lower_bound = calculate_lower_bound(root, set(range(len(jobs))), machines)
    heapq.heappush(heap, (root.lower_bound, root))

    optimal_sequence = None
    while heap:
        _, node = heapq.heappop(heap)
        if len(node.sequence) == len(jobs):
            if optimal_sequence is None or node.time < optimal_sequence.time:
                optimal_sequence = node
            continue

        next_job = find_next_job_sequence(node, set(range(len(jobs))), machines)
        if next_job is None:
            continue

        child = Node(next_job, node.sequence + [next_job], node.time + machines[node.sequence[-1]][next_job], 0)
        child.lower_bound = calculate_lower_bound(child, set(range(len(jobs))), machines)
        if child.lower_bound < node.lower_bound:
            heapq.heappush(heap, (child.lower_bound, child))

    return optimal_sequence.sequence

if __name__ == "__main__":
    # Example jobs and machines setup
    jobs = 3
    machines = np.array([[2, 1, 3],
                         [2, 3, 2],
                         [1, 2, 2]])

    sequence = branch_and_bound(jobs, machines)
    print("Optimal Job Sequence:", sequence)
