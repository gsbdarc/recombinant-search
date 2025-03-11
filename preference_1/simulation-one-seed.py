"""
Simulation for Multivariate Normal Distribution Search

This script performs a simulation to explore points in a multivariate normal distribution space. It utilizes various functionalities like the computation of a preference value, covariance between two points, simulation of the multivariate normal distribution, optimization based on an objective function, and the main search function that integrates these modules.

Key functionalities:

- `preference_func`: Computes a preference value based on the given mean and variance. 
- `cov`: Calculates the covariance between two points.
- `simulate`: Simulates the multivariate normal distribution by computing the conditional mean and conditional variance.
- `objective`: Defines the optimization objective function.
- `find_search_all`: Executes the main search algorithm.

The main loop of the script run a simulation based on input seed value, searching for novel points within the distribution space. Results from simulation are stored in a dictionary which is later saved to a JSON file.

Modules and libraries used:
- `numpy`: For numerical computations.
- `matplotlib`: For plotting (though not explicitly used in this script).
- `scipy`: For optimization and differentiation functionalities.
- `sympy`: For symbolic mathematics.
- `sys`: For system-specific parameters and functions.
- `json`: For handling JSON data.

Usage: call the code with the seed number to run the simulation for
```
python simulation-one-seed.py 0
```

Author: Natalya Rapstine 
Modified: Nov. 8, 2023
"""

import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp

# set random seed
# results will be different for each seed
seed_num = int(sys.argv[1])

def preference_func(mean_, var_):
    """
    This function computes a preference value based on the given mean and variance.
    mean_ and var_ are functions of x, y 
    """
    return mean_ - sp.exp(-3 * mean_ + 4.5 * var_ ** 2)

def cov(point1, point2, staple):
    """
    Covariance between two points.
    point (x, y) is a tuple 
    """

    return min(point1[0], point2[0]) + min(point1[1], point2[1]) + staple[3] * min(point1[0], point2[0]) * min(point1[1], point2[1])


def var_mat(points, staple):
    """
    Var-covariance matrix computation
    points is a list of N tuples
    ith element in points is a tuple of (x, y) coordinates
    Return N x N numpy array of var-cov values 
    """
    
    # preallocate cov matrix
    cov_matrix = np.empty([len(points), len(points)])
    
    if points:
        
        for i in range(len(points)):

            for j in range(len(points)):
                # each point is a tuple and call         
                cov_matrix[i, j] = cov(points[i], points[j], staple)
    
        return cov_matrix
    
    else:
        return np.identity(2)
    
    
def cov_mat(points_list1, points_list2, staple):
    """
    Covariance matrix (generically, not a square matrix) between two lists of points
    points_list1 = [(x0, y0), (x1, y1), (x2, y2)]
    points_list2 = [(i0, j0), (i1, j1)]
    
    Return np array of shape N_points in points_list1 x M_points in points_list2 (len(points_list1) by len(points_list2))
    """
    
    if points_list1 and points_list2:
        
        # preallocate cov matrix 
        cov_matrix = np.empty([len(points_list1), len(points_list2)])
    
        for i in range(len(points_list1)):

            for j in range(len(points_list2)):
                
                cov_matrix[i, j] = cov(points_list1[i], points_list2[j], staple)
                                       
        return cov_matrix
    
    else:
        
        return 0
    
def condit_var(points, history, staple, ivm):
    """
    Conditional variance
    points and history are lists of tuples     
    """
    
    var_matrix = var_mat(points, staple)
    
    cov_term = np.zeros_like(var_matrix)

    if history:
        
        cov_term = np.dot(np.dot(cov_mat(points, history, staple), ivm), np.transpose(cov_mat(points, history, staple)))

    condit_var = var_matrix - cov_term
    
    condit_var = (condit_var + np.transpose(condit_var)) / 2
    
    return condit_var


def mean_vect(points, staple):
    """
    points is a list of tuples (x, y) points [(x0, y0), (x1, y1), (x2, y2), ...] and staple is a list
    
    return list of mean values of length N_points -- same as len(points)
    """
    
    if points:
        return staple[0] + staple[1] * np.array( [p[0] for p in points]) + staple[2] * np.array( [p[1] for p in points])
    
    else:
        return 0.0
    
    
def condit_mean(points, history, outcome, staple, ivm):
    """
    Conditional mean computation
    points and history are lists of tuples 
    outcome is a list with the same length as history and mean_vect(history, staple)
    """
    
    mean_vec = mean_vect(points, staple) # list of len N_points
        
    # if history not empty    
    if history:
        
        cov_term = np.dot(np.dot(cov_mat(points, history, staple), ivm), np.array(outcome) - mean_vect(history, staple))

    else:
        # matrix of zeros with nrows equal to N_points
        cov_term = np.zeros([len(points), ])
             
         
    return mean_vec + cov_term


def adjust_round_error(point, xgrid, ygrid, eps):
    """
    Adjust a point's coordinates to be closer to the nearest grid points (xgrid and ygrid) within a specified tolerance (eps). 
    """
    # Get the dimensions of xgrid and ygrid
    nx = len(xgrid)
    ny = len(ygrid)
    
    # Create a copy of the input point
    newpoint = np.copy(point)
    
    # Iterate through xgrid and check for closeness
    for i in range(nx):
        if abs(newpoint[0] - xgrid[i]) < eps:
            newpoint[0] = xgrid[i]
            break
    
    # Iterate through ygrid and check for closeness
    for j in range(ny):
        if abs(newpoint[1] - ygrid[j]) < eps:
            newpoint[1] = ygrid[j]
            break
    
    return (newpoint[0],  newpoint[1])

def append_grid(point, xgrid, ygrid):
    """
    Append the coordinates of a point to their respective grids (xgrid and ygrid) if they are not already present in those grids.
    """
    # Create copies of xgrid and ygrid to avoid modifying the original lists
    newxgrid = xgrid.copy()
    newygrid = ygrid.copy()
    
    # Check if the x coordinate of point is not in xgrid, and append it if not present
    
    if point[0] not in newxgrid:
        newxgrid.append(point[0])
        newxgrid.sort()  # Sort the updated xgrid
    
    # Check if the y coordinate of point is not in ygrid, and append it if not present
    if point[1] not in newygrid:
        newygrid.append(point[1])
        newygrid.sort()  # Sort the updated ygrid
    
    return newxgrid, newygrid

def new_search_points(point, history, xgrid, ygrid):
    """
    Return new search points based on novelpoint and history
    """
    
    xgrid = np.sort(xgrid)
    ygrid = np.sort(ygrid)

    # calculate the number of elements in xgrid that are less than or equal to the x-coordinate of point 
    nx = np.sum(xgrid <= point[0])
    # calculate the number of elements in ygrid that are less than or equal to the y-coordinate of point. 
    ny = np.sum(ygrid <= point[1])

    # for all the x/y grids below the novel points, make combinations
    newcandidates = []

    for i in range(nx):
        for j in range(ny):
            newcandidates.append((xgrid[i], ygrid[j]))
    
    
    # check if each element in history has x and y coordinates greater than or equal to those of the point. 
    # return a list of elements from history that meet this condition 
    upperhistory = [p for p in history if p[0] >= point[0] and p[1] >= point[1]]
    
    if not upperhistory:
        # if there is no history that is above the novel search, then do nothing. If otherwise, follow the below additional step
        newpoints = list(set(newcandidates) - set(history + [(0, 0)]))
        
    else:
        # select those histories that is above the novel point, including the novel point itself
        # take the x-grid from this upperhistory
        upperxgrid = np.sort(np.unique([p[0] for p in upperhistory]))
        # and the y-grid of upperhistory         
        upperygrid = np.sort(np.unique([p[1] for p in upperhistory]))

        
        # for every possible combination (upperxgrid[i], upperygrid[j]), discard it if it is maximal amongst the upperhistory, and include it if not
        pointsabove = []
        # pointsabove will contain all the points from upperxgrid and upperygrid combinations with 
        for i in range(len(upperxgrid)):
            for j in range(len(upperygrid)):
                
                if not any(p[0] >= upperxgrid[i] and p[1] >= upperygrid[j] for p in upperhistory):
                    pointsabove.append((upperxgrid[i], upperygrid[j]))
       
        # add the above points to the candidate list
        newcandidates = list(set(newcandidates + pointsabove))
        
        newpoints = list(set(newcandidates) - set(history + [(0, 0)]))
       

    return newpoints

# Simulation
def simulate(points, history, outcome, staple, ivm):
    """
    Simulates the multivariate normal distribution by computing the conditional mean and conditional variance
    """
    # Define the mean and covariance matrix for the multivariate normal distribution
    mean = condit_mean(points, history, outcome, staple, ivm)
    
    covariance = condit_var(points, history, staple, ivm)
    
    # random samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(mean, covariance)
    return samples

# Define your optimization objective function
def objective(params):
    """
    Compute the objective value for optimization based on preference function.
    
    This function calculates the objective value for optimization using the preference function. The preference function computes a value based on a given mean and variance which are derived from the conditional mean and variance functions (`condit_mean` and `condit_var`). The objective value is negated for minimization purposes.

    Parameters:
    - params (tuple of float): A tuple containing two values (x, y) which represent the coordinates of a point.

    Returns:
    - float: The negative value of the computed preference for the given point.

    Note:
    The function relies on external variables (history, outcome, staple, and ivm) which must be defined in the scope where this function is called.
    """

    x, y = params

    points = [(x, y)]

    mean = condit_mean(points, history, outcome, staple, ivm)[0]
    var = condit_var(points, history, staple, ivm)[0, 0]

    return -preference_func(mean, var)

def find_search_all(history, outcome, staple, preference_func, ivm, x0, y0):
    """
    Conducts a local search to find the minimum value using the given starting point.

    This function performs a local optimization to determine the minimum value of the objective function using the provided starting point (x0, y0). 
    The optimization process relies on the 'Nelder-Mead' method, appropriate for non-linear objective functions.

    Parameters:
    - history (list): A list of previous points.
    - outcome (list): A list of outcomes associated with the history.
    - staple (list): Parameters for the Brownian staple.
    - preference_func (function): The preference function used for optimization.
    - ivm (matrix): The inverse of the variance-covariance matrix.
    - x0 (float): Initial x-coordinate for the search.
    - y0 (float): Initial y-coordinate for the search.

    Returns:
    - tuple: The function value at the optimal point and the optimal point itself.
    """    
    # x, y >= 0

    bounds = [(0, None), (0, None)]
    
    result = minimize(objective, (x0, y0), method ='Nelder-Mead', bounds = bounds, 
                      options = {'maxiter': 1000})

    return result.fun, result.x

def find_global_minima(history, outcome, staple, preference_func, ivm, initial_points):
    """
    Conducts a global search to find the minimum value over a set of initial points.

    This function iterates over a set of initial points and uses a local search on each point to find the minimum value.
    It then compares all the minimum values and determines the global minimum among them.

    Parameters:
    - history (list): A list of previous points.
    - outcome (list): A list of outcomes associated with the history.
    - staple (list): Parameters for the Brownian staple.
    - preference_func (function): The preference function used for optimization.
    - ivm (matrix): The inverse of the variance-covariance matrix.
    - initial_points (list of tuples): A list of starting points (x, y) for the search.

    Returns:
    - tuple: The maximum function value and its associated (x, y) point. 
    """
    best_value = np.inf  
    best_x = None
    
    for x0, y0 in initial_points:
        value, x = find_search_all(history, outcome, staple, preference_func, ivm, x0, y0)
        if value < best_value:  
            best_value = value
            best_x = x
            
    return best_value, best_x

def generate_initial_points(startx=0.0, endx=1.0, starty=0.0, endy=1.0, num_points=10):
    """
    Generates a grid of initial points in a specified 2D region.

    This function produces a list of evenly spaced points within the specified range of the x and y axes. The number of points along each axis is determined by the num_points parameter.

    Parameters:
    - startx (float): Starting value on the x-axis (default is 0.0).
    - endx (float): Ending value on the x-axis (default is 1.0).
    - starty (float): Starting value on the y-axis (default is 0.0).
    - endy (float): Ending value on the y-axis (default is 1.0).
    - num_points (int): Number of evenly spaced points along each axis (default is 10).

    Returns:
    - list of tuples: List of (x, y) coordinates covering the specified region.
    """
    x_values = np.linspace(startx, endx, num_points)
    y_values = np.linspace(starty, endy, num_points)
    
    return [(x, y) for x in x_values for y in y_values]


def plot_current_state(point, history, derivative_points, iteration, seed_num, save_folder='saved_plots'):
    """
    Visualizes the current iteration of the optimization process.

    This function plots the history of points, the current novel point, and if present, derivative points on a 2D graph. 
    The plotting helps to visualize the evolution and current state of the optimization.

    Parameters:
    - point (tuple): The current novel point as an (x, y) tuple.
    - history (list of tuples): Previously seen points in the search space.
    - derivative_points (list of tuples): Points representing derivative search points, if available.
    - iteration (int): Current iteration number to be displayed in the plot title.

    Returns:
    - None: Saves the plot. 
    """
    save_folder = f"{save_folder}/run-{seed_num:04}"
    
    plt.figure()
    plotting_history = history + [ (0, 0 )]
     
    if derivative_points:
        plt.scatter(*zip(*derivative_points), label='Derivative Points', color='green')
        
    plt.scatter(*zip(*plotting_history), label='History Points', color='black')
    plt.scatter(point[0], point[1], label='Novel Point', color='red')
    plt.title(f'Iter: {iteration:03}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.grid(True)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Save the plot to the specified folder
    filename = os.path.join(save_folder, f'iter_{iteration:03}.jpg')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

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
print(f"Starting simulation for {seed_num:04} seed...")

# simulation with fixed seed
# Parameters for Brownian staple : {psi0, mu1, mu2, sigma0^2}
staple = [0, 0.5, 0.25, 1]

# lists to store histories and outcome
history, novelhistory, outcome, derivative_history = [], [], [], []
    
# x, y grids
xgrid = [0]
ygrid = [0]
  
# error tolerance 
eps = 10 ** (-4)
 
# max iter number 
maxiter = 500

# Main dictionary to store all the results
sim_results = {}

cond_two, cond_three = [], []

for i in range(maxiter):
    print(f"i: {i}")
    # fix the inverse of var - covar matrix first
    ivm = np.linalg.inv(var_mat(history, staple))
    
    min_x = min(xgrid)
    min_y = min(ygrid)
    max_x = max(xgrid)
    max_y = max(ygrid)
    
    if min_x == max_x:
        max_x = min_x + 1
    else:
        max_x += 1
        
    if min_y == max_y:
        max_y = min_y + 1
    else:
        max_y += 1
        
    # run optimization 100 times with different initial points and choose the optima
    initial_points = generate_initial_points(startx=min_x, endx=max_x, starty=min_y, endy=max_y)
    novelpoint = find_global_minima(history, outcome, staple, preference_func, ivm, initial_points)[1]
    novelpoint = adjust_round_error(novelpoint, xgrid, ygrid, eps)
    
    # check if novelpoint is already in history 
    if novelpoint in history + [(0, 0)]:
        repeat_point = novelpoint
        break
        
    else:
        
        novelhistory.append(novelpoint)
        xgrid, ygrid = append_grid(novelpoint, xgrid, ygrid)
        
        # Check for CONDITION 2: at each iter, x and y are in the grey regions
        # Maximal x and y explored in the history
        max_hist_x = max([p[0] for p in history + [(0, 0)]], default=float('-inf'))
        max_hist_y = max([p[1] for p in history + [(0, 0)]], default=float('-inf'))

        # (i), (ii), and (iii)        
        if not (
            (novelpoint[0] >= max_hist_x) or 
            (novelpoint[1] >= max_hist_y) or 
            (
                (novelpoint[0] < max_hist_x and novelpoint[1] < max_hist_y) and 
                all(not(p[0] >= novelpoint[0] and p[1] >= novelpoint[1]) for p in history)
            )
        ):
            cond_two.append(False)
            print(f'Condition 2 failed for iteration {i}')
        else:
            cond_two.append(True)
        
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
        derivative_points = [p for p in searchpoints if p not in set(history + [novelpoint])]
        
        # Uncomment to make plots of the search path
        # plot_current_state(novelpoint, history, derivative_points, i, seed_num)
        history.extend(searchpoints)
        if derivative_points:
            derivative_history.extend(derivative_points)
        
    # max outcome at which point
    max_value = max(outcome)
    max_index = outcome.index(max_value)

# Save results 
# save repeat point
# save max_value and point for max_value outcome
    
# CONDITION 1: check that repeat point has max_value outcome from all history points
    
# Create the dictionary to be saved as JSON
# Store the results for this seed in the main dictionary
sim_results[seed_num] = {
    "history": history,
    "novelhistory": novelhistory,
    "outcome": outcome,
    "derivative_history": derivative_history,
    "repeat_point": repeat_point,
    "max_outcome": max_value,
    "max_point": history[max_index],
    "cond_two": cond_two,
    "cond_three": cond_three
}

 
print(f"Finished running simulation for seed {seed_num}.")
print(f"sim_results: {sim_results}")

# End timing
end_time = time.time()

# Print out the total execution time
print(f"Seed {seed_num} simulation executed in {end_time - start_time:0.2f} seconds")

# write out results
# Check if the directory exists, if not, create it
if not os.path.exists(f"results"):
    os.makedirs(f"results")

filename = f"results/run-{seed_num:04}.json"

with open(filename, 'w') as outfile:
    json.dump(sim_results, outfile, cls=NumpyEncoder, indent=4)
