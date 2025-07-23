from vrp_parser import *
import nevergrad as ng
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def decode_solution_with_capacity(encoded_solution: np.ndarray, instance, num_vehicles: int) -> List[List[int]]:
    """
    Converts a float vector to a list of vehicle routes using a greedy capacity based split
    """
    sorted_customers = np.argsort(encoded_solution)

    sorted_customers = sorted_customers + 1 # +1 to skip depot index 0

    routes = []
    current_route = []
    current_demand = 0.0
    vehicle_count = 0

    for cust_id in sorted_customers:
        demand = instance.demands[cust_id]
        
        if current_demand + demand > instance.capacity:
            routes.append(current_route)
            vehicle_count += 1

            if vehicle_count >= num_vehicles:
                routes[-1].extend(sorted_customers[len(routes)*len(current_route):])
                return routes

            current_route = []
            current_demand = 0.0

        current_route.append(cust_id)
        current_demand += demand

    if current_route:
        routes.append(current_route)

    while len(routes) < num_vehicles:
        routes.append([])

    return routes


def vrp_objective(encoded_solution: np.ndarray, instance: VRPInstance, 
                 num_vehicles: int, penalty_weight: float = 1000.0) -> float:
    """
    Compute total cost of VRP solution with penalty for constraint violations
    """
    #routes = decode_solution(encoded_solution, num_vehicles)
    routes = decode_solution_with_capacity(encoded_solution, instance, num_vehicles)
    total_distance = 0.0
    total_penalty = 0.0
    
    for route in routes:
        if not route:
            continue
            
        current = 0 #starting at depot index 0
        route_demand = 0
        
        for cust_id in route:
            route_demand += instance.demands[cust_id]
            total_distance += instance.distance_matrix[current][cust_id]
            current = cust_id
        
        # Return to depot
        total_distance += instance.distance_matrix[current][0]
        
        # Capacity constraint
        if route_demand > instance.capacity:
            total_penalty += (route_demand - instance.capacity)
    
    return total_distance + penalty_weight * total_penalty

def solve_vrp(instance, num_vehicles=25, budget=10000, optimizer_name="GeneticDE"):
    num_customers = len(instance.demands) - 1
    
    parametrization = ng.p.Array(shape=(num_customers,), lower=0, upper=1)
    optimizer_cls = ng.optimizers.registry[optimizer_name]
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget)

    best_cost = float('inf')
    best_vector = None
    best_routes = []

    for i in range(budget):
        candidate = optimizer.ask()
        vector = candidate.value
        cost = vrp_objective(vector, instance, num_vehicles)
        optimizer.tell(candidate, cost)

        if cost < best_cost:
            best_cost = cost
            best_vector = vector
            best_routes = decode_solution_with_capacity(vector, instance, num_vehicles)

    return best_cost, best_routes


# 1. Loading instances
# instance = load_vrp_file("X-n101-k25.vrp")
# num_customers = len(instance.demands) - 1  # Dont include depot
# num_vehicles = 13  # k25 file name

# print(solve_vrp(instance=instance, num_vehicles=25, budget=1000000, optimizer_name="GeneticDE"))


##### BEFORE THE FUNCTION solve_vrp
# # 2. Just testing
# print("\nInstance verification:")
# print(f"Depot index: {instance.depot_index}")
# print(f"Customer IDs in demands: {sorted(instance.demands.keys())}")
# print(f"Distance matrix shape: {len(instance.distance_matrix)}x{len(instance.distance_matrix[0])}")

# # 3. Nevergrad setup
# parametrization = ng.p.Array(shape=(num_customers,), lower=0, upper=1)
# optimizer = ng.optimizers.OnePlusOne(
#     parametrization=parametrization,
#     budget=10000 
# )

# # 4. Test with a random solution first
# print("\nTesting with random solution:")
# test_vector = np.random.rand(num_customers)
# test_cost = vrp_objective(test_vector, instance, num_vehicles)
# print(f"Test cost: {test_cost}") 

# # 5. Run optimization
# print("\nRunning optimization...")
# objective = lambda x: vrp_objective(x, instance, num_vehicles)
# best_cost = float('inf')
# best_vector = None
# best_routes = []

# for i in range(1000000): 
#     candidate = optimizer.ask()
#     vector = candidate.value
#     cost = vrp_objective(vector, instance, num_vehicles)
    
#     optimizer.tell(candidate, cost)

#     if cost < best_cost:
#         best_cost = cost
#         best_vector = vector
#         best_routes = decode_solution_with_capacity(vector, instance, num_vehicles)
        
#         print(f"Iteration {i+1}, New Best Cost: {best_cost:.2f}")
#         for v_idx, route in enumerate(best_routes):
#             route_1_based = [c + 1 for c in route]
#             print(f"  Route #{v_idx+1}: {' '.join(map(str, route_1_based))}")
#         print("="*40)

###########################

######### BEFORE MULTIPLE ITERATIONS
#recommendation = optimizer.minimize(objective)

# 6. Get the best solution
# best_encoded = recommendation.value
# best_routes = decode_solution(best_encoded, num_vehicles)
# best_cost = vrp_objective(best_encoded, instance, num_vehicles)

# print(f"\nBest solution found with cost: {best_cost}")
# print("Sample routes (0-based indices):")
# for i, route in enumerate(best_routes): 
#     print(f"Vehicle {i+1}: {route}")



# test_vector = np.random.rand(len(instance.demands) - 1)  # exclude depot
# routes = decode_solution_with_capacity(test_vector, instance, num_vehicles=num_vehicles)
# cost = vrp_objective(recommendation.value, instance, num_vehicles)
# print("Cost: ", cost)
# for i, route in enumerate(routes):
#     route_1_based = [cust + 1 for cust in route]
#     print(f"Vehicle {i+1}: {route_1_based}")


def plot_routes(instance: VRPInstance, routes: List[List[int]], best_cost):
    plt.figure(figsize=(10, 8))
    
    depot_x, depot_y = instance.node_coords[instance.depot_index]

    plt.scatter(depot_x, depot_y, c='red', marker='s', label='Depot')

    for idx, route in enumerate(routes):
        if not route:
            continue
        
        x = [depot_x]
        y = [depot_y]
        
        for cust_id in route:
            cx, cy = instance.node_coords[cust_id + 1] 
            x.append(cx)
            y.append(cy)
        
        x.append(depot_x)
        y.append(depot_y)
        
        plt.plot(x, y, label=f"Vehicle {idx+1}")
        plt.scatter(x[1:-1], y[1:-1])  

    plt.title(f"Total cost: {best_cost:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot_routes(instance, best_routes)

def plot_routes_streamlit(instance: VRPInstance, routes: List[List[int]], best_cost):
    fig, ax = plt.subplots(figsize=(10, 8))

    depot_x, depot_y = instance.node_coords[instance.depot_index]
    ax.scatter(depot_x, depot_y, c='red', marker='s', label='Depot')

    for idx, route in enumerate(routes):
        if not route:
            continue

        x = [depot_x]
        y = [depot_y]

        for cust_id in route:
            cx, cy = instance.node_coords[cust_id + 1]
            x.append(cx)
            y.append(cy)

        x.append(depot_x)
        y.append(depot_y)

        ax.plot(x, y, label=f"Vehicle {idx+1}")
        ax.scatter(x[1:-1], y[1:-1])  # Customers

    ax.set_title(f"Total cost: {best_cost:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)

    return fig
