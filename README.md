This repository contains Python implementations for various multi-tier allocation optimization problems. The solutions focus on minimizing costs, improving service levels, and reducing environmental impact in supply chain and logistics scenarios.

Overview
The project includes models for optimizing allocations in:
1. Single-Tier Allocation: Assigning couriers to delivery zones to minimize travel distances.
2. Two-Tier Allocation: Allocating products between warehouses and retailers to minimize transportation costs.
3. Complex Multi-Tier Allocation: Managing allocations across suppliers, warehouses, and retailers while balancing costs, service levels, and environmental considerations.

Features
1. Linear programming-based optimization using scipy.optimize.linprog.
2. Customizable constraints for supply, demand, fleet capacity, and service-level agreements.
3. Visualization of allocation results using heatmaps.

File Structure
1. Multi-Tier Allocation Optimization.py: Main Python script containing:
  1.1 Problem definitions and data setup.
  1.2 Optimization model implementation.
  1.3 Visualization of results.
2. Input Data: Embedded within the script for quick testing.
3. Output: Optimized allocations and minimized costs, displayed as matrices and heatmaps.


Getting Started:
Prerequisites
1. Python 3.8 or higher.
2. Libraries: numpy, pandas, scipy, matplotlib, seaborn.

Installation
Clone the repository: git clone https://github.com/your-username/multi-tier-allocation-optimization.git
Navigate to the repository: cd multi-tier-allocation-optimization
Install dependencies: pip install -r requirements.txt

Usage
1. Run the script: python Multi-Tier\ Allocation\ Optimization.py
2. View the console output for optimized allocations and total costs.
3. Check the heatmap visualizations for a graphical representation of the allocations.


Scenarios Addressed:
1. Courier Allocation
  Objective: Minimize total travel distance while meeting courier demand in each delivery zone.
  Constraints:
    Each zone receives the exact number of couriers required.
    Each courier is assigned to one zone only.
2. Warehouse-Retailer Allocation
  Objective: Minimize transportation costs while meeting retailer demands and adhering to warehouse capacities.
  Output: Optimal allocation matrix and minimized total transportation cost.
3. Multi-Tier Optimization
  Objective: Optimize supplier-to-warehouse and warehouse-to-retailer allocations while:
  Minimizing total costs (transportation + inventory).
  Maximizing service levels.
  Reducing carbon footprint.

Visualization:
The repository includes heatmaps to visualize:
1. Courier allocation to delivery zones.
2. Warehouse-to-retailer product flow.
3. Multi-tier supplier-to-retailer allocations.


Contributing: Contributions are welcome! Feel free to open issues, submit PRs, or suggest improvements.

License: This project is licensed under the MIT License.
