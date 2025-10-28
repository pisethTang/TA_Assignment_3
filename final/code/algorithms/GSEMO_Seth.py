import ioh 
from .algorithm_interface import Algorithm
import numpy as np



class GSEMO_Seth(Algorithm):
    '''
    Global Simple Evolutionary Multi-Objective Optimizer (GSEMO) algorithm implementation.
    '''
    def __init__(self, budget: int,
                 name: str = "GSEMO_Seth",
                 algorithm_info: str = "Global Simple Evolutionary Multi-Objective Optimizer (GSEMO) algorithm implementation."):
        super().__init__(budget, name=name, algorithm_info=algorithm_info)



    def evaluate_individual_multiobjective_fitness(self, individual: np.ndarray, func: ioh.problem.GraphProblem) -> np.ndarray:
        '''
            Evaluates the multi-objective fitness of a given individual using the provided function.
        '''

        # objective 1: score (to be maximized)
        score = func(individual.tolist())

        if score < 0:
            objective1_score = -np.inf # to ensure this solution won't dominate any other solution
        else:
            objective1_score = score
        
        # objective 2: cost (to be minimized)
        # since the cost is assumed to be uniform (basically constant), that is,
        # the cost of each edge is 1, then the cost of an individual is simply the number of 1s in its bitstring representation 
        # i.e. cost(individual) = sum(individual[i] * cost(edge_i)) = sum(individual[i] * 1) = sum(individual)

        cost = np.sum(individual)  # number of 1s in the bitstring
        objective2_cost = cost

        return (objective1_score, objective2_cost)

    def dominates(self, objective_a: tuple[int, int], objective_b: tuple[int, int]) -> bool:
        """
        Check if individual a (with objective values objective_a) dominates individual b (with objective values objective_b).
        Assumes a maximization problem for the first objective and minimization for the second.
        Returns True if a dominates b, False otherwise.
        """
        score_a, cost_a = objective_a
        score_b, cost_b = objective_b

        # condition 0 (ensures that both individuals are not identical)
        are_identical: bool = (score_a == score_b) and (cost_a == cost_b)
        if are_identical:
            return False

        # condition 1: a is no worse than b in all objectives
        no_worse: bool = (score_a >= score_b) and (cost_a <= cost_b)

        # condition 2: a is strictly better than b in at least one objective
        strictly_better: bool = (score_a > score_b) or (cost_a < cost_b)

        return no_worse and strictly_better


    def run_optimization_loop(self, func: ioh.problem.GraphProblem) -> list[tuple[np.ndarray, int]]:
        """"Runs the core GSEMO logic and returns the final population (the Pareto front)."""
        n = func.meta_data.n_variables # Get dimension from the problem

        # --- Initialization ---
        # 1. Choose x uniformly at random
        x = np.random.randint(2, size=n) 
        # 2. Determine g(x)
        x_fitness = self.evaluate_individual_multiobjective_fitness(x, func) 
        # 3. P <- {x} (Store as pairs: (individual, fitness))
        population = [(x, x_fitness)] 

        # --- Main Loop ---
        # 11. until stop (budget condition)
        while func.state.evaluations < self.budget: 
            # 5. Choose x from P uniformly at random
            parent_idx = np.random.randint(len(population))
            parent, _ = population[parent_idx] # Unpack parent and its fitness

            # 6. Create x' by flipping each bit with prob 1/n
            offspring = parent.copy() 
            mutation_mask = np.random.rand(n) < (1 / n)
            offspring[mutation_mask] = 1 - offspring[mutation_mask]

            # Check if offspring is identical to parent to avoid re-evaluation
            if np.array_equal(offspring, parent):
                continue # Skip if mutation didn't change anything

            # 7. Determine g(x')
            offspring_fitness = self.evaluate_individual_multiobjective_fitness(offspring, func)
            
            # --- Population Update Logic ---
            # 8. Check if x' is dominated by any z in P
            is_offspring_dominated = False
            for _, individual_fitness in population: # Loop through stored fitness values
                if self.dominates(individual_fitness, offspring_fitness):
                    is_offspring_dominated = True
                    break 

            # 9. If x' is not dominated, include x' into P
            if not is_offspring_dominated:
                # Add the new offspring (and its fitness)
                population.append((offspring, offspring_fitness))
                
                # 10. Delete all z from P dominated by x'
                # Filter the list, keeping only those NOT dominated by the new offspring
                population = [(ind, fit) for ind, fit in population 
                            if not self.dominates(offspring_fitness, fit)]
        return population


    def __call__(self, func: ioh.problem.GraphProblem): # Use base Problem class
        '''
        Method called by IOHexperimenter for logging and running the algorithm on a given problem instance.
        func: An instance of ioh.problem.GraphProblem representing the optimization problem.
        '''

        # Run the optimization loop
        self.run_optimization_loop(func)

