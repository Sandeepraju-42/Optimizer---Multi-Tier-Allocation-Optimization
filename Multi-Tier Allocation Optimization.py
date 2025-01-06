# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:04:08 2024

@author: sande

Multi-Tier Allocation Optimization

assumptions:
1 We have three delivery zones with a set demand for couriers.
2 A pool of couriers is available in each zone.
3 The goal is to allocate couriers to meet demand across the zones with minimal 
total travel distance.

Explanation of the Code
Data Setup: 
    We define zones and couriers, where each courier has distances to each zone 
    (e.g., Zone A, Zone B, Zone C).

Cost Matrix: 
    A flattened array of travel distances to minimize total travel distance 
    across all assignments.

Constraints:
1 Equality Constraint (A_eq): Ensures that each zone gets the exact demand met 
by couriers.
2 Inequality Constraint (A_ub): Ensures each courier can only be allocated to 
one zone.

Objective: 
    The objective function minimizes the total travel distance using linear programming
    
Output:
    The output provides an optimized allocation of couriers to zones that 
    minimizes the travel distance. The minimum total travel distance is also 
    displayed, allowing Wolt to understand the efficiency gain achieved through 
    optimized courier allocation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Define the data
zones = ['Zone_A', 'Zone_B', 'Zone_C']
couriers = {'Courier_1': [1, 3, 5],  # Distances from Courier_1 to Zone_A, Zone_B, Zone_C
            'Courier_2': [2, 4, 3],
            'Courier_3': [3, 1, 2],
            'Courier_4': [4, 2, 4]}

demand = [3, 4, 3]  # Demand for couriers in each zone

# Convert courier data to a cost matrix (flattened for linprog)
cost_matrix = np.array([couriers[courier] for courier in couriers]).flatten()

# Constraints
# Each zone should get the exact demand met
A_eq = np.zeros((len(zones), len(couriers) * len(zones)))

# Fill in the equality constraints for each zone
for i, zone in enumerate(zones):
    for j in range(len(couriers)):
        A_eq[i, i + j * len(zones)] = 1

# Each courier can only be allocated to one zone
A_ub = np.zeros((len(couriers), len(couriers) * len(zones)))
for i in range(len(couriers)):
    for j in range(len(zones)):
        A_ub[i, i * len(zones) + j] = 1

# Objective
b_eq = demand  # demand for each zone
b_ub = [1] * len(couriers)  # each courier is used once

# Linear programming to minimize total travel distance
result = linprog(cost_matrix, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')

# Output the results
allocations = result.x.reshape(len(couriers), len(zones))
allocation_df = pd.DataFrame(allocations, index=couriers.keys(), columns=zones)
print("Optimal Courier Allocations to Minimize Travel Distance:")
print(allocation_df)

print("\nMinimum Total Travel Distance:", result.fun)


"""
#########################################################
Optimal Allocation Across Multiple Tiers
Problem: A company has to allocate products across a multi-tier distribution network 
(e.g., suppliers, warehouses, and retailers). 

The objective:
    minimize transportation costs while satisfying demand across all tiers.

Solution Approach
We'll solve this as a linear programming problem with constraints on supply and 
demand across each tier.


We set up a linear programming problem where c represents the flattened 
transportation cost matrix from each warehouse to each retailer.

Constraints are created to ensure that the supply limits at each warehouse 
(A_supply and b_supply) and the demand at each retailer (A_demand and b_demand) 
are respected.

We solve this problem using linprog from SciPy with the "highs" method, designed 
for large-scale linear programming problems.

The output shows the optimal product allocation from each warehouse to each retailer, 
minimizing total transportation costs.

#########################################################
"""

import numpy as np
from scipy.optimize import linprog
import seaborn as sns
import matplotlib.pyplot as plt


# Demand and supply for each tier (warehouses to retailers)
demand = [200, 300, 250]  # Demand at retailers
supply = [300, 500]       # Supply from warehouses

# Transportation costs from each warehouse to each retailer
costs = [[2, 3, 1], [5, 4, 2]]

# Define bounds for each variable (cannot supply negative quantity)
bounds = [(0, None) for _ in range(len(costs) * len(costs[0]))]

# Create objective function (flatten costs matrix)
c = np.array(costs).flatten()

# Create inequality constraints for supply (each warehouse has a limit)
A_supply = np.zeros((len(supply), len(c)))
for i in range(len(supply)):
    A_supply[i, i * len(demand):(i + 1) * len(demand)] = 1
b_supply = supply

# Create equality constraints for demand (each retailer has a required demand)
A_demand = np.zeros((len(demand), len(c)))
for j in range(len(demand)):
    A_demand[j, j::len(demand)] = 1
b_demand = demand

# Solve the problem using linear programming
result = linprog(c, A_ub=A_supply, b_ub=b_supply, A_eq=A_demand, b_eq=b_demand, bounds=bounds, method="highs")

# Reshape solution to show allocation from each warehouse to each retailer
allocation = result.x.reshape((len(supply), len(demand)))
print("Optimal Allocation (Warehouses to Retailers):")
print(allocation)
print("Total Transportation Cost:", result.fun)

"""
This matrix shows how much product each warehouse should supply to each retailer:

For example, Warehouse 1 supplies 200 units to Retailer 1, 100 units to Retailer 2, and 0 units to Retailer 3.
Similarly, Warehouse 2 supplies 0 units to Retailer 1, 200 units to Retailer 2, and 250 units to Retailer 3.

Total Transportation Cost: 2000.0, indicating the minimized transportation cost given the allocations.
"""

#Viz
#creating a heatmap of the allocation matrix
# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(allocation, annot=True, cmap="YlGnBu", fmt=".0f",
            xticklabels=["Retailer 1", "Retailer 2", "Retailer 3"],
            yticklabels=["Warehouse 1", "Warehouse 2"])
plt.title("Optimal Allocation from Warehouses to Retailers")
plt.xlabel("Retailers")
plt.ylabel("Warehouses")
plt.show()


"""
#########################################################
A Complex Multi-Tier Allocation Optimization

Problem Statement: 
    In this complex scenario, you manage inventory across multiple tiers—suppliers, 
    warehouses, and retail locations. The goal is to allocate shipments from 
    suppliers to warehouses and then from warehouses to retailers 
    
optimizing for:
    1 Minimizing Total Cost: This includes transportation costs and inventory 
        holding costs at each tier.
    2 Maximizing Service Level: Service level is defined as the ability to meet 
        retailer demand within a specified time frame.
    3 Minimizing Environmental Impact: This considers the carbon footprint of 
        transportation routes between tiers.
        
Each warehouse has a maximum capacity, and each retailer has demand that must be 
met within certain lead times. Furthermore, certain warehouses are closer to 
specific retailers, allowing for faster delivery.

Solution Approach:

Define Objectives:
    Objective 1: Minimize total transportation and inventory costs.
    Objective 2: Maximize service level (e.g., by meeting demand within the lead time).
    Objective 3: Minimize environmental impact from transportation.

Constraints:
    Demand must be met at each retailer.
    Maximum warehouse capacity must not be exceeded.
    Certain shipments must be prioritized for faster routes.

Python Implementation: 
    We’ll define an MTAO problem where we optimize allocation across suppliers, 
    warehouses, and retailers with a focus on balancing cost, service level, 
    and carbon impact.
#########################################################
"""
# Libraries
# ----------------------------
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------------

# ----------------------------
# Example data for a 3-Tier MTAO problem
# ----------------------------

# Define costs, capacities, demands, CO2 impacts, and SLAs for each leg.

# --- Supplier to Warehouse (Suppliers: S1, S2; Warehouses: W1, W2)
supplier_to_warehouse_cost = np.array([[3, 2], [4, 3]])  # Transport cost matrix (S1 -> W1/W2, S2 -> W1/W2)
supplier_capacities = [400, 300]  # Supplier capacities
warehouse_demand_for_suppliers = [500, 200]  # Demand at each warehouse
supplier_carbon_impact = np.array([[0.2, 0.3], [0.3, 0.25]])  # CO2 impact per unit
supplier_service_levels = np.array([[0.95, 0.9], [0.9, 0.93]])  # SLA levels

# --- Warehouse to Retailer (Warehouses: W1, W2; Retailers: R1, R2, R3)
warehouse_to_retailer_cost = np.array([[2, 3, 1], [4, 2, 5]])  # Transport cost matrix (W1/W2 -> R1, R2, R3)
warehouse_capacities = [500, 600]  # Warehouse capacities
retailer_demands = [200, 250, 150]  # Retailer demands
warehouse_carbon_impact = np.array([[0.3, 0.4, 0.2], [0.6, 0.3, 0.5]])  # CO2 impact per unit
warehouse_service_levels = np.array([[0.9, 0.8, 0.95], [0.85, 0.92, 0.88]])  # SLA levels
# ----------------------------

# ----------------------------
# Construct the combined objective for the 3-Tier problem
# ----------------------------

# Flattened cost vector combining both legs of the allocation
c = np.concatenate([
    (supplier_to_warehouse_cost + supplier_carbon_impact - supplier_service_levels).flatten(),  # Supplier-Warehouse
    (warehouse_to_retailer_cost + warehouse_carbon_impact - warehouse_service_levels).flatten()  # Warehouse-Retailer
])

# Constraints: Building matrices for each tier
# --- Supplier to Warehouse Constraints
A_eq_sw = np.zeros((len(warehouse_demand_for_suppliers), supplier_to_warehouse_cost.size))
for i in range(len(warehouse_demand_for_suppliers)):
    A_eq_sw[i, i::len(warehouse_demand_for_suppliers)] = 1
b_eq_sw = warehouse_demand_for_suppliers

# --- Warehouse to Retailer Constraints
A_eq_wr = np.zeros((len(retailer_demands), warehouse_to_retailer_cost.size))
for i in range(len(retailer_demands)):
    A_eq_wr[i, i::len(retailer_demands)] = 1
b_eq_wr = retailer_demands

# Combine constraints and bounds
A_eq = np.block([
    [A_eq_sw, np.zeros((A_eq_sw.shape[0], A_eq_wr.shape[1]))],  # Supplier-to-Warehouse constraints
    [np.zeros((A_eq_wr.shape[0], A_eq_sw.shape[1])), A_eq_wr]   # Warehouse-to-Retailer constraints
])
b_eq = np.concatenate([b_eq_sw, b_eq_wr])


# Define bounds for each allocation leg
bounds = [(0, cap) for cap in np.concatenate([np.ravel([supplier_capacities]*len(warehouse_demand_for_suppliers)), 
                                              np.ravel([warehouse_capacities]*len(retailer_demands))])]

# Solve the problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
# Reshape solution for easier interpretation
allocation_sw = result.x[:len(supplier_to_warehouse_cost.flatten())].reshape(len(supplier_capacities), len(warehouse_demand_for_suppliers))
allocation_wr = result.x[len(supplier_to_warehouse_cost.flatten()):].reshape(len(warehouse_capacities), len(retailer_demands))

# ----------------------------
# Output the solution
# ----------------------------
print("Optimal Allocation (Supplier to Warehouse):")
print(allocation_sw)
print("Optimal Allocation (Warehouse to Retailer):")
print(allocation_wr)
print("Total Combined Objective:", result.fun)

# ----------------------------
# Visualization: Allocation Heatmap
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Supplier to Warehouse Allocation Heatmap
sns.heatmap(allocation_sw, annot=True, cmap="YlGnBu", fmt=".0f",
            xticklabels=["Warehouse 1", "Warehouse 2"],
            yticklabels=["Supplier 1", "Supplier 2"], ax=axes[0])
axes[0].set_title("Supplier to Warehouse Allocation")
axes[0].set_xlabel("Warehouses")
axes[0].set_ylabel("Suppliers")

# Warehouse to Retailer Allocation Heatmap
sns.heatmap(allocation_wr, annot=True, cmap="YlGnBu", fmt=".0f",
            xticklabels=["Retailer 1", "Retailer 2", "Retailer 3"],
            yticklabels=["Warehouse 1", "Warehouse 2"], ax=axes[1])
axes[1].set_title("Warehouse to Retailer Allocation")
axes[1].set_xlabel("Retailers")
axes[1].set_ylabel("Warehouses")

plt.tight_layout()
plt.show()
















# Libraries
# ----------------------------
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------------

# ----------------------------
# Example data for a 3-Tier MTAO problem with Fleet Size, Routes, and Utilization
# ----------------------------

# Define costs, capacities, demands, CO2 impacts, SLAs for each leg.
supplier_to_warehouse_cost = np.array([[3, 2], [4, 3]])  # Transport cost matrix (S1 -> W1/W2, S2 -> W1/W2)
supplier_capacities = [400, 300]  # Supplier capacities
warehouse_demand_for_suppliers = [500, 200]  # Demand at each warehouse
supplier_carbon_impact = np.array([[0.2, 0.3], [0.3, 0.25]])  # CO2 impact per unit
supplier_service_levels = np.array([[0.95, 0.9], [0.9, 0.93]])  # SLA levels

warehouse_to_retailer_cost = np.array([[2, 3, 1], [4, 2, 5]])  # Transport cost matrix (W1/W2 -> R1, R2, R3)
warehouse_capacities = [500, 600]  # Warehouse capacities
retailer_demands = [200, 250, 150]  # Retailer demands
warehouse_carbon_impact = np.array([[0.3, 0.4, 0.2], [0.6, 0.3, 0.5]])  # CO2 impact per unit
warehouse_service_levels = np.array([[0.9, 0.8, 0.95], [0.85, 0.92, 0.88]])  # SLA levels

# Define fleet parameters
fleet_capacity = 1000  # Total fleet capacity for each route
num_vehicles = 3  # Number of vehicles available

# Decision Variables (1 for using a route, 0 for not using it)
# For simplicity, we'll assume each supplier-warehouse-retailer combination is a possible route.
routes = np.array([[1, 0, 1], [0, 1, 0]])  # Route availability (can be extended)

# ----------------------------
# Construct the combined objective for the 3-Tier problem
# ----------------------------

# Flattened cost vector combining both legs of the allocation
c = np.concatenate([ 
    (supplier_to_warehouse_cost + supplier_carbon_impact - supplier_service_levels).flatten(),  # Supplier-Warehouse
    (warehouse_to_retailer_cost + warehouse_carbon_impact - warehouse_service_levels).flatten(),  # Warehouse-Retailer
    np.zeros(len(routes.flatten()))  # Route-related variables (added as zero cost for simplicity)
])

# Constraints: Building matrices for each tier
# --- Supplier to Warehouse Constraints
A_eq_sw = np.zeros((len(warehouse_demand_for_suppliers), supplier_to_warehouse_cost.size))
for i in range(len(warehouse_demand_for_suppliers)):
    A_eq_sw[i, i::len(warehouse_demand_for_suppliers)] = 1
b_eq_sw = warehouse_demand_for_suppliers

# --- Warehouse to Retailer Constraints
A_eq_wr = np.zeros((len(retailer_demands), warehouse_to_retailer_cost.size))
for i in range(len(retailer_demands)):
    A_eq_wr[i, i::len(retailer_demands)] = 1
b_eq_wr = retailer_demands

# --- Fleet Capacity Constraints (Each route must not exceed fleet capacity)
A_fleet_capacity = np.zeros((routes.size, len(c)))
for i in range(routes.shape[0]):  # Iterate over each route
    for j in range(routes.shape[1]):  # For each possible route
        route_index = i * routes.shape[1] + j
        A_fleet_capacity[i, route_index] = 1  # Add constraint for route utilization
b_fleet_capacity = np.full(routes.shape[0], fleet_capacity)  # Set fleet capacity limits for each route

# --- Vehicle Limit Constraints (We can only use a total of 'num_vehicles' vehicles)
A_vehicle_limit = np.zeros((1, len(c)))
for i in range(routes.shape[0]):
    for j in range(routes.shape[1]):
        route_index = i * routes.shape[1] + j
        A_vehicle_limit[0, route_index] = 1  # Sum over all routes to count the vehicles
b_vehicle_limit = np.array([num_vehicles])  # Only num_vehicles can be used

# Total number of variables (decision variables)
num_variables = len(c)

# Combine all equality constraints
A_eq = np.block([
    [A_eq_sw, np.zeros((A_eq_sw.shape[0], A_eq_wr.shape[1])), np.zeros((A_eq_sw.shape[0], A_fleet_capacity.shape[1]))],  # Supplier-to-Warehouse constraints
    [np.zeros((A_eq_wr.shape[0], A_eq_sw.shape[1])), A_eq_wr, np.zeros((A_eq_wr.shape[0], A_fleet_capacity.shape[1]))],  # Warehouse-to-Retailer constraints
    [A_fleet_capacity, np.zeros((A_fleet_capacity.shape[0], A_eq_sw.shape[1] + A_eq_wr.shape[1]))],  # Fleet capacity constraints
    [A_vehicle_limit, np.zeros((A_vehicle_limit.shape[0], A_eq_sw.shape[1] + A_eq_wr.shape[1] + A_fleet_capacity.shape[1]))]  # Vehicle limit constraints
])
b_eq = np.concatenate([b_eq_sw, b_eq_wr, b_fleet_capacity, b_vehicle_limit])

# Define bounds for each allocation leg
bounds = [(0, cap) for cap in np.concatenate([np.ravel([supplier_capacities]*len(warehouse_demand_for_suppliers)), 
                                              np.ravel([warehouse_capacities]*len(retailer_demands)), 
                                              np.ravel([1]*len(routes.flatten()))])]

# Solve the problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
allocation_sw = result.x[:len(supplier_to_warehouse_cost.flatten())].reshape(len(supplier_capacities), len(warehouse_demand_for_suppliers))
allocation_wr = result.x[len(supplier_to_warehouse_cost.flatten()):len(supplier_to_warehouse_cost.flatten()) + len(warehouse_to_retailer_cost.flatten())].reshape(len(warehouse_capacities), len(retailer_demands))

# ----------------------------
# Output the solution
# ----------------------------
print("Optimal Allocation (Supplier to Warehouse):")
print(allocation_sw)
print("Optimal Allocation (Warehouse to Retailer):")
print(allocation_wr)
print("Total Combined Objective:", result.fun)

# ----------------------------
# Visualization: Allocation Heatmap
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Supplier to Warehouse Allocation Heatmap
sns.heatmap(allocation_sw, annot=True, cmap="YlGnBu", fmt=".0f",
            xticklabels=["Warehouse 1", "Warehouse 2"],
            yticklabels=["Supplier 1", "Supplier 2"], ax=axes[0])
axes[0].set_title("Supplier to Warehouse Allocation")
axes[0].set_xlabel("Warehouses")
axes[0].set_ylabel("Suppliers")

# Warehouse to Retailer Allocation Heatmap
sns.heatmap(allocation_wr, annot=True, cmap="YlGnBu", fmt=".0f",
            xticklabels=["Retailer 1", "Retailer 2", "Retailer 3"],
            yticklabels=["Warehouse 1", "Warehouse 2"], ax=axes[1])
axes[1].set_title("Warehouse to Retailer Allocation")
axes[1].set_xlabel("Retailers")
axes[1].set_ylabel("Warehouses")

plt.tight_layout()
plt.show()




# --- Combine all equality constraints with appropriate padding
A_eq = np.block([
    [A_eq_sw, np.zeros((A_eq_sw.shape[0], A_eq_wr.shape[1])), np.zeros((A_eq_sw.shape[0], A_fleet_capacity.shape[1]))],  # Supplier-to-Warehouse constraints
    [np.zeros((A_eq_wr.shape[0], A_eq_sw.shape[1])), A_eq_wr, np.zeros((A_eq_wr.shape[0], A_fleet_capacity.shape[1]))],  # Warehouse-to-Retailer constraints
    [A_fleet_capacity, np.zeros((A_fleet_capacity.shape[0], A_eq_sw.shape[1] + A_eq_wr.shape[1]))],  # Fleet capacity constraints
    [A_vehicle_limit, np.zeros((A_vehicle_limit.shape[0], A_eq_sw.shape[1] + A_eq_wr.shape[1] + A_fleet_capacity.shape[1]))]  # Vehicle limit constraints
])

# Verify that the number of columns in the block matrix matches the length of c
assert A_eq.shape[1] == num_variables, "The number of columns in A_eq does not match the length of the decision variable vector."
