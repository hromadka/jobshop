class Node:
    def __init__(self, value, level, i, j, path, cost):
        self.value = value
        self.level = level
        self.i = i
        self.j = j
        self.path = path
        self.cost = cost

def calculate_cost(matrix, path):
    cost = 0
    for row, col in path:
        cost += matrix[row][col]
    return cost

def find_minimum_cost(matrix):
    n = len(matrix)
    total_nodes = 0

    # Initialize the root node
    root = Node(value=0, level=0, i=-1, j=-1, path=[], cost=0)
    priority_queue = [root]

    while priority_queue:
        node = priority_queue.pop(0)

        # If all jobs are assigned
        if node.level == n - 1:
            return node.path

        # Go to the next level
        for j in range(n):
            if j not in node.path:
                child = Node(value=0, level=node.level + 1, i=node.level, j=j,
                             path=node.path + [(node.level, j)], cost=node.cost + matrix[node.level][j])

                # Bound
                child.value = calculate_cost(matrix, child.path)

                # Add child to the queue
                priority_queue.append(child)
                total_nodes += 1

        # Sort the queue based on the lower bound
        priority_queue.sort(key=lambda x: x.value)

    return None

def print_solution(matrix, path):
    print("Assigned Jobs:")
    for i, j in path:
        print(f"Worker {i} -> Job {j} (Cost: {matrix[i][j]})")
    print("Total Cost:", calculate_cost(matrix, path))

# Example usage
matrix = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]

solution = find_minimum_cost(matrix)
if solution:
    print_solution(matrix, solution)
else:
    print("No feasible solution found.")
