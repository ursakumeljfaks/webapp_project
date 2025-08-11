from itertools import cycle
import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib as mpl
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from vrp_parser import *


vrp_file_path = "./X-n101-k25.vrp" 
vrp_instance = load_vrp_file(vrp_file_path)

num_nodes = len(vrp_instance.node_coords)
capacity = int(vrp_instance.capacity)
n_vehicles = 26 # change depending on the test problem
N = num_nodes

# preparing demands 
demands_list = [0] * N
for node_id, demand_value in vrp_instance.demands.items():
    index = vrp_instance.node_id_to_index[node_id]
    demands_list[index] = int(demand_value)
demands = list(map(int, demands_list))

# preparing distances
distances = np.array(vrp_instance.distance_matrix)
distances = np.round(distances, decimals=4)
coordinates = pd.DataFrame([vrp_instance.node_coords[node_id] for node_id in sorted(vrp_instance.node_coords.keys())], columns=['x', 'y']) # for plotting

#######################################

## model
depot_node_id = vrp_instance.depot_index
depot_node_index = vrp_instance.node_id_to_index[depot_node_id]
manager = pywrapcp.RoutingIndexManager(
    N, n_vehicles, depot_node_index
)

# create Routing model
routing = pywrapcp.RoutingModel(manager)

## parameters -> single function for the distance callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    if from_node < 0 or to_node < 0 or from_node >= N or to_node >= N:
        return 0
    return int(distances[from_node][to_node])

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    if node < 0 or node >= N:
        return 0
    return int(demands[node])

demands_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

## constraints
routing.AddDimensionWithVehicleCapacity(
    demands_callback_index,
    0,  # null capacity slack
    [capacity,] * n_vehicles,  # vehicle maximum capacities 
    True,  # start cumul to zero
    'Capacity'
)


## objective
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

## solution
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC # or CHRISTOFIDES but this one provided better solutions
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.FromSeconds(5)

## solving the problem
solution = routing.SolveWithParameters(search_parameters)

##########################################################

if solution:
    print(f"Objective value: {solution.ObjectiveValue()}")
    tours = []
    for vehicle_id in range(n_vehicles):
        index = routing.Start(vehicle_id)
        tours.append([])
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            tours[-1].append(node_index)
            index = solution.Value(routing.NextVar(index))
        else:
            node_index = manager.IndexToNode(index)
            tours[-1].append(node_index)

    print(tours)
    
    # plotting the solution
    cmap = mpl.colormaps["tab20"]
    colors = cycle(cmap.colors)

    fig, ax = plt.subplots(figsize=[6.5, 5], dpi=100)
    for r, tour in enumerate(tours):
        if len(tour) > 1: # only plot routes with more than just the depot
            c = next(colors)
            t = np.array(tour)
            x = coordinates.values[t, 0]
            y = coordinates.values[t, 1]
            ax.scatter(x, y, color=c, label=f"R{r}")
            ax.plot(x, y, color=c)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02))
    fig.tight_layout()
    plt.show()

else:
    print("No solution found!")

