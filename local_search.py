import numpy as np


def cost_function(solution):
    # Compute the cost of the given solution
    total_cost = sum([e_i for e_i in solution])
    return total_cost


def neighbor_function(solution):
    # Generate a list of neighboring solutions
    neighbors = []
    for i in range(len(solution)):
        neighbor = list(solution)
        neighbor[i] += np.random.normal(0, 0.1)
        neighbors.append(neighbor)
    return neighbors


def local_search(initial_solution, neighbor_function, max_iterations):
    current_solution = initial_solution
    current_cost = cost_function(current_solution)
    for i in range(max_iterations):
        neighbors = neighbor_function(current_solution)
        best_neighbor = min(neighbors, key=lambda x: cost_function(x))
        best_cost = cost_function(best_neighbor)
        if best_cost < current_cost:
            current_solution = best_neighbor
            current_cost = best_cost
        else:
            return current_solution, current_cost
    return current_solution, current_cost
