from .algorithm_interface import Algorithm
import ioh
import numpy as np


# class OnePlusOneEA(Algorithm):
#     def __init__(self, budget: int):
#         super().__init__(budget, name="(1+1)_EA", algorithm_info="(1+1) Evolutionary Algorithm.")

#     def __call__(self, problem: ioh.problem.PBO):
#         # (1+1) EA implementation (not including the external loop for multiple runs)
#         n = problem.meta_data.n_variables
#         # Initialize a random solution
#         current = np.random.randint(0, 2, size=n)
#         current_fitness = problem(current.tolist())
        
#         mutation_prob: float = 1/n


#         while problem.state.evaluations < self.budget:
#             if problem.state.optimum_found: # break early if optimum is found
#                 break

#             mutation_mask = np.random.rand(n) < mutation_prob # vectorized mutation

#             # only flips when at least one bit is selected for mutation
#             if not mutation_mask.any():
#                 continue
            

#             # apply mutation in-place
#             offspring = current.copy()
#             # flip chosen bits 
#             offspring[mutation_mask] = 1 - offspring[mutation_mask]

#             offspring_fitness = problem(offspring.tolist())

#             if offspring_fitness >= current_fitness:
#                 current = offspring
#                 current_fitness = offspring_fitness



class OnePlusOneEA(Algorithm):
    def __call__(self, problem: ioh.problem.PBO):
        n = problem.meta_data.n_variables
        current = np.random.randint(0, 2, size=n)
        current_fitness = problem(current.tolist())
        
        mutation_prob = 1.0 / n
        
        while problem.state.evaluations < self.budget:
            if problem.state.optimum_found:
                break
            
            # Sample number of bits to flip from binomial
            num_flips = np.random.binomial(n, mutation_prob)
            
            # Ensure at least 1 flip (conditional binomial)
            if num_flips == 0:
                num_flips = 1
            
            # Randomly select which bits to flip
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            
            # Create offspring
            offspring = current.copy()
            offspring[flip_indices] = 1 - offspring[flip_indices]
            
            offspring_fitness = problem(offspring.tolist())
            
            if offspring_fitness >= current_fitness:
                current = offspring
                current_fitness = offspring_fitness