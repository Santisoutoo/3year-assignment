import numpy as np

class ParticleSwarmOptimization(object):
    """
    Particle Swarm Optimization (PSO) implementation.

    PSO is a population-based stochastic optimization technique inspired by 
    social behavior of bird flocking or fish schooling.
    """

    def __init__(
        self,
        objective_function: callable,
        n_particles: int = 20,
        dimensions: int = 2,
        bounds: tuple[float, float] = (0, 5),
        w: float = 0.8,
        c1: float = 0.1,
        c2: float = 0.1,
        seed: int = 100
    ):
        """
        Initialize the PSO algorithm.

        Parameters:
        -----------
        objective_function : function
            The function to be minimized
        n_particles : int
            Number of particles in the swarm
        dimensions : int
            Number of dimensions in the search space
        bounds : tuple
            (min, max) bounds for each dimension
        w : float
            Inertia weight - controls influence of previous velocity
        c1 : float
            Cognitive parameter - controls attraction to personal best
        c2 : float
            Social parameter - controls attraction to global best
        seed : int
            Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Initialize particles' positions and velocities
        # Positions are uniformly distributed within bounds
        self.X = np.random.rand(dimensions, n_particles) * \
            (bounds[1] - bounds[0]) + bounds[0]
        # Initial velocities are small random values
        self.V = np.random.randn(dimensions, n_particles) * 0.1

        # Initialize personal best positions and objective values
        # Initially, personal best is the starting position
        self.pbest = self.X.copy()
        # Evaluate all particles
        self.pbest_obj = self.evaluate(self.X)

        # Initialize global best position and objective value
        # Find particle with best objective value
        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

        # Store optimization history
        self.history = {
            'positions': [self.X.copy()],
            'velocities': [self.V.copy()],
            'pbest': [self.pbest.copy()],
            'gbest': [self.gbest.copy()],
            'gbest_obj': [self.gbest_obj]
        }

    def evaluate(self, X):
        """
        Evaluate the objective function for all particles.

        Parameters:
        -----------
        X : ndarray
            Particles' positions

        Returns:
        --------
        ndarray
            Objective function values for all particles
        """
        return self.objective_function(X[0], X[1])

    def update(self):
        """
        Perform one iteration of the PSO algorithm.

        Steps:
        1. Update velocities based on inertia, cognitive, and social components
        2. Update positions based on velocities
        3. Evaluate new positions
        4. Update personal and global bests

        Returns:
        --------
        dict
            Current state of the swarm
        """
        # Generate random coefficients for stochastic behavior
        r1, r2 = np.random.rand(2)

        # Update velocities
        # Components: inertia + cognitive + social
        self.V = (self.w * self.V +                                   # Inertia component
                  # Cognitive component
                  self.c1 * r1 * (self.pbest - self.X) +
                  self.c2 * r2 * (self.gbest.reshape(-1, 1) - self.X))  # Social component

        # Update positions by adding velocities
        self.X = self.X + self.V

        # Evaluate new positions
        obj = self.evaluate(self.X)

        # Update personal best positions and objective values
        # Only update if new position is better (lower objective value)
        mask = (self.pbest_obj >= obj)
        self.pbest[:, mask] = self.X[:, mask]
        self.pbest_obj = np.minimum(self.pbest_obj, obj)

        # Update global best position and objective value
        min_idx = self.pbest_obj.argmin()
        self.gbest = self.pbest[:, min_idx]
        self.gbest_obj = self.pbest_obj[min_idx]

        # Store current state for history
        self.history['positions'].append(self.X.copy())
        self.history['velocities'].append(self.V.copy())
        self.history['pbest'].append(self.pbest.copy())
        self.history['gbest'].append(self.gbest.copy())
        self.history['gbest_obj'].append(self.gbest_obj)

        # Return current state for visualization
        return {
            'X': self.X,
            'V': self.V,
            'pbest': self.pbest,
            'gbest': self.gbest,
            'gbest_obj': self.gbest_obj
        }

    def optimize(self, iterations=50):
        """
        Run the PSO algorithm for a specified number of iterations.

        Parameters:
        -----------
        iterations : int
            Number of iterations to run

        Returns:
        --------
        tuple
            (global best position, global best objective value)
        """
        for _ in range(iterations):
            self.update()

        return self.gbest, self.gbest_obj

    def get_history(self):
        """
        Get the optimization history.

        Returns:
        --------
        dict
            Optimization history containing positions, velocities, 
            personal bests, global best, and objective values
        """
        return self.history
