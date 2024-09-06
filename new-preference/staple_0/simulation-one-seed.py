"""
Simulation for Multivariate Normal Distribution Search with a new preference function.

This script performs a simulation to explore points in a multivariate normal distribution space. It utilizes various functionalities like the computation of a preference value, covariance between two points, simulation of the multivariate normal distribution, optimization based on an objective function, and the main search function that integrates these modules.

The main loop of the script run a simulation based on input seed value, searching for novel points within the distribution space. Results from simulation are stored in a dictionary which is later saved to a JSON file.

Modules and libraries used:
- `numpy`: For numerical computations.

- `matplotlib`: For plotting. 
- `scipy`: For optimization and differentiation functionalities.
- `sympy`: For symbolic mathematics.
- `sys`: For system-specific parameters and functions.
- `json`: For handling JSON data.

Usage: call the code with the seed number to run the simulation for
```
python simulation-one-seed.py 0
```

Author: Natalya Rapstine 
Modified: Apr. 3, 2024
"""

import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp
from simulation_utils import *

# set random seed
# results will be different for each seed

seed_num =int(sys.argv[1])

class NumpyEncoder(json.JSONEncoder):
    """
    Need to convert any numpy to write out results in JSON format
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
          
# Start timing
start_time = time.time()

# save results into a dictionary as JSON
sim_results = {}

repeat_point = None

np.random.seed(seed_num)
print(f"Starting simulation for {seed_num:05} seed...")

# simulation with fixed seed
# Parameters for Brownian staple : {psi0, mu1, mu2, sigma0^2}
# change this for psi0 < 0, mu1 > 0, mu2 > 0 :
staple = [-0.6, 1, 0.99, 0.5]

# lists to store histories and outcome
history, novelhistory, outcome, derivative_history, frontier = [], [], [], [], []
    
# x, y grids
xgrid = [0]
ygrid = [0]
  
# error tolerance 
eps = 10 ** (-4)
 
# max iter number 
maxiter = 500

# Main dictionary to store all the results
sim_results = {}

cond_three = []

for i in range(maxiter):
    print(f"i: {i}")
    # fix the inverse of var - covar matrix first
    ivm = np.linalg.inv(var_mat(history, staple))
    
    min_x = min(xgrid)
    min_y = min(ygrid)
    max_x = max(xgrid)
    max_y = max(ygrid)
    
    if min_x == max_x:
        max_x = min_x + 0.3
    else:
        max_x += 0.3
        
    if min_y == max_y:
        max_y = min_y + 0.3
    else:
        max_y += 0.3
#    print(f'min_x: {min_x}') 
#    print(f'max_x: {max_x}') 
#    print(f'min_y: {min_y}') 
#    print(f'max_y: {max_y}')
 
    # run optimization with different initial points and choose the optima
    initial_points = generate_initial_points(startx=min_x, endx=max_x, starty=min_y, endy=max_y)
    novelpoint = find_global_minima(history, outcome, staple, preference_func, ivm, initial_points)[1]
    novelpoint = adjust_round_error(novelpoint, xgrid, ygrid, eps)
    
    #print(f"novelpoint: {novelpoint}")
    
    # check if novelpoint is already in history 
    if novelpoint in history + [(0, 0)]:
        repeat_point = novelpoint
        #print(f"repeat_point: {repeat_point}")
        break
        
    else:
        # exercise 3: check if novelpoint (point in red) is in the frontier of all the past points (in black) -- history
        print(f'froniter status: {is_frontier(novelpoint, history)}')
        frontier.append(is_frontier(novelpoint, history))

        novelhistory.append(novelpoint)
        xgrid, ygrid = append_grid(novelpoint, xgrid, ygrid)
        
        # Check for CONDITION 3: at each iter, novel point has x or y that coincides with x-grid or y-grid in history
        # Extracting x's and y's
        grid_history_x = set( [p[0] for p in history + [(0, 0)]] )
        grid_history_y = set( [p[1] for p in history + [(0, 0)]])

        # check that the novel point's x-entry or its y-entry coincides with that of a point already in the history:
        # novelpoint's x or y must be in the grid history x or y sets
        if (novelpoint[0] in grid_history_x) or (novelpoint[1] in grid_history_y):
            cond_three.append(True)
        else:
            cond_three.append(False)
 
        searchpoints = new_search_points(novelpoint, history, xgrid, ygrid)
        newoutcome = simulate(searchpoints, history, outcome, staple, ivm)
        outcome.extend(newoutcome)
        #print(f"outcome: {outcome}")
        derivative_points = [p for p in searchpoints if p not in set(history + [novelpoint])]
        
        # Uncomment to make plots of the search path
        # plot_current_state(novelpoint, history, derivative_points, i, seed_num)
        history.extend(searchpoints)
        if derivative_points:
            derivative_history.extend(derivative_points)
        
# there is a repeat point, so now we check CONDITION 1
# Find the minimum absolute difference from 0 for all points' outcome
abs_differences = [abs(o - 0.0) for o in outcome]
# Find min difference
min_difference = min(abs_differences)
print(f"min_difference: {min_difference}")
# min_difference_index = abs_differences.index(min_difference)
# print(f"min_difference_index: {min_difference_index}")

# Find outcome at repeat point
if repeat_point != (0.0, 0.0):
    repeat_point_index = history.index(repeat_point)
    print(f"repeat_point_index: {repeat_point_index}")
    repeat_point_outcome = outcome[repeat_point_index]
    print(f"repeat_point_outcome: {repeat_point_outcome}")

    # Calculate the absolute difference of the repeat point's outcome from 0
    repeat_point_abs_difference = abs(repeat_point_outcome - 0.0)

    # Check if the repeat point's outcome is the closest to 0
    if repeat_point_abs_difference == min_difference:
        # print("The repeat point's outcome is the closest to 0 among all outcomes.")
        print("CONDITION 1 PASSED")
        cond_one = True
    else:
        print("CONDITION 1 FAILED! There is another point with an outcome closer to 0 than the repeat point's outcome.")
        cond_one = False
else:
    # if repeat point is (0, 0), outcome is 0     
    print(f"repeat_point: {repeat_point}")
    cond_one = True

# Exercise 1:
positive_points = [(x, y) for x, y in history if x > 0 and y > 0]
if positive_points:
    exercise_one = 'moves to the combinatoric phase'

else:
    exercise_one = 'remains in the field phase'


# Exercise 3:
n_switches = count_switches(frontier)

# Save results 
sim_results[seed_num] = {
        "history": history,
        "novelhistory": novelhistory,
        "outcome": outcome,
        "derivative_history": derivative_history,
        "repeat_point": repeat_point,  
        "cond_one": cond_one,
        "cond_three": cond_three,
        "exercise_one": exercise_one,
        "frontier": frontier,
        "switches": n_switches
    }

print(f"Finished running simulation for seed {seed_num}.")
print(f"sim_results: {sim_results}")

# End timing
end_time = time.time()

# Print out the total execution time
print(f"Seed {seed_num} simulation executed in {end_time - start_time:0.2f} seconds")

# write out results
# Check if the directory exists, if not, create it
results_dir = f"results"
if not os.path.exists(f"{results_dir}"):
    os.makedirs(f"{results_dir}")

filename = f"{results_dir}/run-{seed_num:05}.json"

with open(filename, 'w') as outfile:
    json.dump(sim_results, outfile, cls=NumpyEncoder, indent=4)
