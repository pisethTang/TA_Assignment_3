from .algorithm_interface import Algorithm
import ioh
import numpy as np

class RandomizedLocalSearch(Algorithm):
    def __init__(self, budget: int):
        super().__init__(budget, name="Randomized Local Search", algorithm_info="Randomized Local Search Algorithm.")

    def __call__(self, problem: ioh.problem.PBO) -> None:
        # Randomised Local Search implementation (not including the external loop for multiple runs)
        current_sol = np.random.randint(0,2 , size=problem.meta_data.n_variables)
        current_fitness = problem(current_sol.tolist())


        for _ in range(self.budget):
            if problem.state.optimum_found: # break early if optimum is found
                break



            # create a neighbor by flipping one random bit
            flip_index = np.random.randint(0, problem.meta_data.n_variables)
            current_sol[flip_index] = 1 - current_sol[flip_index]  # Flip a random bit
            neighbor_fitness = problem(current_sol.tolist())

            # if the neighbor is better or equal, replace current solution
            if neighbor_fitness >= current_fitness:
                current_fitness = neighbor_fitness
            else: 
                # undo the flip 
                current_sol[flip_index] = 1 - current_sol[flip_index]



