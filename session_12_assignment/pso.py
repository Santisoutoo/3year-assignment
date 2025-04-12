# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: f1_strat_manager
#     language: python
#     name: python3
# ---

# # PSO Implementation

# ---
#

# ### 1. Import Libraries

# ---

import numpy as np
from pso_class import ParticleSwarmOptimization  


# ---

# ### 2. Define the Objective Function

def f(x, y):
    """
    Objective function with multiple local minima.
    
    This function combines quadratic terms and sinusoidal components to create
    a complex landscape with multiple local minima and one global minimum.
    
    Parameters:
    -----------
    x, y : array-like
        Input coordinates
        
    Returns:
    --------
    array-like
        Function values at the given coordinates
    """
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)



# ---

# ### 3. Compute and visualize the function landscape 

print("1. Global Minimum Calculation")
# Create 1D arrays for x and y values between 0 and 5
x_vals = np.linspace(0, 5, 1000)
y_vals = np.linspace(0, 5, 1000)

# Create a 2D grid of x and y values
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

# Compute the function values for each (x, y) pair on the grid
z_grid = f(x_grid, y_grid)

# Find the global minimum value in the grid
min_idx = np.argmin(z_grid)
y_min, x_min = np.unravel_index(min_idx, z_grid.shape)

# Convert grid indices back to x and y coordinates
x_min, y_min = x_vals[x_min], y_vals[y_min]

# Compute the minimum function value at the computed coordinates
min_value = f(x_min, y_min)


print(f"   Global minimum found at x={x_min:.6f}, y={y_min:.6f}")
print(f"   Minimum value: f(x,y)={min_value:.6f}")

# ---

# ### 4. Initizalide the PSO Algorithm

n_particles =     20        # Number of particles in the swarm
w =               0.8       # Inertia weight
c1 = c2 =         0.1       # Cognitive and social parameters
bounds =          (0, 5)    # Search space bounds
seed =            100       # Random seed for reproducibility
iterations =      50        # Number of iterations to run

print("\n2. PSO Initialization and Execution")
print(f"   Number of particles: {n_particles}")
print(f"   Parameters: w={w}, c1={c1}, c2={c2}")
print(f"   Search space bounds: {bounds}")
print(f"   Number of iterations: {iterations}")

# Initialize the Particle Swarm Optimization (PSO) algorithm with the specified parameters.
pso = ParticleSwarmOptimization(
    objective_function = f,    # Function to optimize
    n_particles = n_particles, # Number of particles in the swarm
    dimensions = 2,            # Working in 2D
    bounds = bounds,           # Boundaries for the search space
    w = w,                     # Inertia weight
    c1 = c1,                   # Cognitive (self) coefficient
    c2 = c2,                   # Social (neighbor) coefficient
    seed = seed                # Seed for reproducibility
)

# Run the optimizer for a given number of iterations.
# This returns the best position (gbest) and its corresponding objective function value (gbest_obj).
gbest, gbest_obj = pso.optimize(iterations)


# ---

# ### 5. Display results

print("\n4. Results:")
print(f"   PSO found best solution at:")
print(f"      x = {gbest[0]:.6f}, y = {gbest[1]:.6f}")
print(f"      f(x,y) = {gbest_obj:.6f}")
print(f"\n   True global minimum:")
print(f"      x = {x_min:.6f}, y = {y_min:.6f}")
print(f"      f(x,y) = {min_value:.6f}")
# Calculate the Euclidean distance error between the best found position (gbest) and the known minimum point.
#
# Here, 'gbest' is assumed to be a list or array where:
#   - gbest[0] is the x-coordinate of the best solution found by the PSO algorithm.
#   - gbest[1] is the y-coordinate of the best solution found by the PSO algorithm.
#
# 'x_min' and 'y_min' represent the x and y coordinates of the actual (or expected) minimum point.
distance_error = np.sqrt((gbest[0] - x_min)**2 + (gbest[1] - y_min)**2)
value_error = abs(gbest_obj - min_value)
print(f"\n   Error Analysis:")
print(f"      Distance error: {distance_error:.6f}")
print(f"      Function value error: {value_error:.6f}")

# ---

# ### 6. Convergence Analysis

# Print a header to indicate the beginning of the convergence analysis.
print("\n5. Convergence Analysis")  

# Retrieve the history of the PSO algorithm's execution.
# The method pso.get_history() is expected to return a dictionary containing various metrics
# recorded at each iteration of the optimization process.
history = pso.get_history()  

# Extract the list of global best objective values from the history.
# The key 'gbest_obj' contains the best objective (fitness) value found at every iteration.
gbest_values = history['gbest_obj']  

# Print a message to indicate that the following output will list the objective values
# obtained at selected iterations.
print("   Objective value by iteration:")

# Determine which iterations to display.
# If there are at least 50 iterations, we choose specific iterations: 0, 4, 9, 19, and 49.
# Otherwise, generate a list of indices spread uniformly through the available iterations.
iterations_to_show = (
    [0, 4, 9, 19, 49] 
    if len(gbest_values) >= 50 
    else list(range(0, len(gbest_values), max(1, len(gbest_values)//5)))
)

# Loop through the selected iteration indices.
for i in iterations_to_show:
    # Safety check: ensure that the iteration index is within the range of available values.
    if i < len(gbest_values):
        # Print the iteration number and the corresponding objective value.
        # The value is formatted to six decimal places for clarity.
        print(f"      Iteration {i}: {gbest_values[i]:.6f}")

# Check if there is more than one objective value available in the history to proceed with improvement analysis.
if len(gbest_values) > 1:
    # Calculate the total improvement: the difference between the objective value at the first iteration
    # and the final objective value (last iteration).
    total_improvement = gbest_values[0] - gbest_values[-1]
    
    # Compute the percentage improvement relative to the initial objective value.
    improvement_percent = (total_improvement / gbest_values[0]) * 100
    
    # Print the total improvement and the corresponding percentage.
    print(f"\n   Total improvement: {total_improvement:.6f} ({improvement_percent:.2f}%)")
    
    # Now, check if the algorithm has converged based on recent improvements.
    # Only proceed if there are at least 10 iterations to evaluate recent changes.
    if len(gbest_values) >= 10:
        # Calculate the improvement over the last 10 iterations.
        # This is the difference between the objective value from 10 iterations ago and the final value.
        recent_improvement = gbest_values[-10] - gbest_values[-1]
        
        # Determine if the improvement is minimal:
        # If the improvement is less than 1e-6, we assume that the solution has converged.
        if recent_improvement < 1e-6:
            print("   Convergence status: Solution has converged (minimal improvement in last 10 iterations)")
        else:
            # Otherwise, if there's still a meaningful improvement, indicate that the solution is still improving.
            print("   Convergence status: Solution still improving")
    else:
        # If there are not enough iterations to evaluate the recent improvement (fewer than 10 iterations),
        # print a message indicating that the convergence status cannot be determined.
        print("   Convergence status: Not enough iterations to determine")


# ---

# # PSO Algorithm Results Analysis
#
# ## Comparison of Results
#
# | Metric | PSO Solution | True Global Minimum | Difference |
# |--------|-------------|---------------------|------------|
# | x-coordinate | 3.185418 | 3.183183 | 0.002235 |
# | y-coordinate | 3.129725 | 3.128128 | 0.001597 |
# | Function value | -1.808352 | -1.808306 | 0.000046 |
# | Euclidean distance | - | - | 0.002746 |
#
# ## Convergence Analysis
#
# | Iteration | Objective Value | Improvement from Previous |
# |-----------|----------------|--------------------------|
# | 0 | 0.940058 | - |
# | 4 | -0.219024 | 1.159082 |
# | 9 | -1.764420 | 1.545396 |
# | 19 | -1.808225 | 0.043805 |
# | 49 | -1.808352 | 0.000127 |
#
# Total improvement: 2.748410 (292.37%)
#
# ## Conclusions
#
# 1. **Accuracy**: The PSO algorithm successfully found the global minimum of the non-linear function with high precision. The difference between the PSO solution and the true global minimum is negligible (function value error of only 0.000046).
#
# 2. **Convergence**: The algorithm demonstrated excellent convergence properties:
#    - Rapid improvement in early iterations (particularly between iterations 0-9)
#    - Fine-tuning in middle iterations (9-19)
#    - Minimal improvement in later iterations, indicating proper convergence
#
# 3. **Efficiency**: The algorithm converged to a near-optimal solution within just 20 iterations, with only minimal improvements thereafter.
#
# 4. **Stability**: The solution stabilized in the later iterations, confirming that the algorithm successfully settled at the global minimum rather than oscillating between local minima.
#
# 5. **Parameter Effectiveness**: The chosen parameters (w=0.8, c1=c2=0.1, n_particles=20) proved very effective for this particular optimization problem, balancing exploration and exploitation.
