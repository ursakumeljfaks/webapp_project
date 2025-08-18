import streamlit as st
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

def solve_vrp(vrp_instance, n_vehicles, time_limit=5):
    num_nodes = len(vrp_instance.node_coords)
    capacity = int(vrp_instance.capacity)
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
    
    # Model setup
    depot_node_id = vrp_instance.depot_index
    depot_node_index = vrp_instance.node_id_to_index[depot_node_id]
    manager = pywrapcp.RoutingIndexManager(N, n_vehicles, depot_node_index)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node < 0 or to_node < 0 or from_node >= N or to_node >= N:
            return 0
        return int(distances[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Demand callback
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        if node < 0 or node >= N:
            return 0
        return int(demands[node])

    demands_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Add capacity constraint
    routing.AddDimensionWithVehicleCapacity(
        demands_callback_index,
        0,  # null capacity slack
        [capacity,] * n_vehicles,  # vehicle maximum capacities 
        True,  # start cumul to zero
        'Capacity'
    )

    # Set objective
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        tours = []
        total_distance = 0
        for vehicle_id in range(n_vehicles):
            index = routing.Start(vehicle_id)
            tours.append([])
            route_distance = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                tours[-1].append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            tours[-1].append(node_index)
            total_distance += route_distance
        
        return total_distance, tours
    else:
        return None, None

def plot_routes_streamlit(vrp_instance, tours):
    coordinates = pd.DataFrame([vrp_instance.node_coords[node_id] for node_id in sorted(vrp_instance.node_coords.keys())], 
                             columns=['x', 'y'])
    
    cmap = mpl.colormaps["tab20"]
    colors = cycle(cmap.colors)

    fig, ax = plt.subplots(figsize=[8, 6], dpi=100)
    for r, tour in enumerate(tours):
        if len(tour) > 1:  # only plot routes with more than just the depot
            c = next(colors)
            t = np.array(tour)
            x = coordinates.values[t, 0]
            y = coordinates.values[t, 1]
            ax.scatter(x, y, color=c, label=f"Route {r+1}")
            ax.plot(x, y, color=c)
    
    # Highlight depot
    depot_index = vrp_instance.node_id_to_index[vrp_instance.depot_index]
    ax.scatter(coordinates.values[depot_index, 0], 
               coordinates.values[depot_index, 1], 
               color='red', marker='s', s=100, label='Depot')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02))
    fig.tight_layout()
    return fig

