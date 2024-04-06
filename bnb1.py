# bnb example from ChatGPT 3.5 - 3/8/2024
# prompt = please write a python program that implements branch and bound to solve an assignment problem.  
# Please do not use any optimization libraries.  I would like to see a pure python implementation.

import numpy as np

class Node:
    def __init__(self, value=None, parent=None, level=0, cost=0, assigned=None, reduced_matrix=None):
        self.value = value
        self.parent = parent
        self.level = level
        self.cost = cost
        self.assigned = assigned
        self.reduced_matrix = reduced_matrix

def row_reduction(matrix):
    # Reduce rows
    reduced_matrix = np.copy(matrix)
    for i in range(len(matrix)):
        min_val = min(reduced_matrix[i])
        if min_val != 0 and min_val != float('inf'):
            reduced_matrix[i] -= min_val
    return reduced_matrix

def col_reduction(matrix):
    # Reduce columns
    reduced_matrix = np.copy(matrix)
    for j in range(len(matrix[0])):
        min_val = min(reduced_matrix[:, j])
        if min_val != 0 and min_val != float('inf'):
            reduced_matrix[:, j] -= min_val
    return reduced_matrix

def calculate_cost(assigned, matrix):
    cost = 0
    for row, col in assigned:
        cost += matrix[row, col]
    return cost

def bound(node, matrix):
    reduced_matrix = np.copy(node.reduced_matrix)
    cost = node.cost

    # Row reduction
    reduced_matrix = row_reduction(reduced_matrix)

    # Column reduction
    reduced_matrix = col_reduction(reduced_matrix)

    # Add the reduced cost to the cost so far
    cost += np.sum(reduced_matrix)

    return cost

def assign_job(node, matrix, row, col):
    assigned = node.assigned.copy()
    assigned.append((row, col))

    reduced_matrix = np.copy(node.reduced_matrix)
    reduced_matrix[row, :] = float('inf')  # Mark row as unavailable
    reduced_matrix[:, col] = float('inf')  # Mark column as unavailable

    return Node(value=(row, col), parent=node, level=node.level + 1,
                cost=calculate_cost(assigned, matrix), assigned=assigned,
                reduced_matrix=reduced_matrix)

def branch_and_bound(matrix):
    n = len(matrix)
    min_cost = float('inf')
    min_assignment = None

    # Initialize root node
    root = Node(reduced_matrix=matrix)

    queue = [root]

    while queue:
        current_node = queue.pop(0)

        # If all jobs are assigned
        if current_node.level == n:
            if current_node.cost < min_cost:
                min_cost = current_node.cost
                min_assignment = current_node.assigned
            continue

        # Branching for each job
        for j in range(n):
            # Check if the job is not already assigned
            if all(j != job[1] for job in current_node.assigned):
                child_node = assign_job(current_node, matrix, current_node.level, j)
                child_node.cost = bound(child_node, matrix)

                # Add child node to the queue only if its cost is promising
                if child_node.cost < min_cost:
                    queue.append(child_node)

    return min_assignment

if __name__ == "__main__":
    # Example cost matrix
    cost_matrix = np.array([[5, 3, 0],
                            [2, 6, 1],
                            [0, 0, 4]])

    assignment = branch_and_bound(cost_matrix)
    print("Optimal Assignment:", assignment)
