from .algorithm_interface import Algorithm
import ioh
import numpy as np


class OnePlusOneEA(Algorithm):
    def __init__(self, budget: int):
        super().__init__(budget, name="(1+1)_EA", algorithm_info="(1+1) Evolutionary Algorithm.")

    def __call__(self, problem: ioh.problem.PBO):
        # (1+1) EA implementation (not including the external loop for multiple runs)
        n = problem.meta_data.n_variables
        # Initialize a random solution
        current = np.random.randint(0, 2, size=n)
        current_fitness = problem(current.tolist())
        
        mutation_prob: float = 1/n


        while problem.state.evaluations < self.budget:
            if problem.state.optimum_found: # break early if optimum is found
                break

            mutation_mask = np.random.rand(n) < mutation_prob # vectorized mutation

            # only flips when at least one bit is selected for mutation
            if not mutation_mask.any():
                continue
            

            # apply mutation in-place
            offspring = current.copy()
            # flip chosen bits 
            offspring[mutation_mask] = 1 - offspring[mutation_mask]

            offspring_fitness = problem(offspring.tolist())

            if offspring_fitness >= current_fitness:
                current = offspring
                current_fitness = offspring_fitness


