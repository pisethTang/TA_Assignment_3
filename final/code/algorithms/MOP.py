from .algorithm_interface import Algorithm
import numpy as np
import ioh
from typing import Callable, Optional, Tuple

class MOP (Algorithm):
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 K_Elites: int,
                 mutation_prob: Optional[float] = None,
                 name: Optional[str] = "MOP", 
                 algorithm_info: str = "MOP population-based optimisation",
                ):
        super().__init__(budget, name, algorithm_info)
        self.rng = np.random.default_rng()  # Added RNG for random operations
        self.pop_size = pop_size
        self.K_Elites = K_Elites
        self.mutation_prob = mutation_prob

    def _Repair(self, X: np.ndarray, problem: ioh.problem.GraphProblem) -> None:
        for index, x in enumerate(X):
            fitness = problem(x.tolist())
            if fitness >= 0:
                continue

            ones_index = np.flatnonzero(X[index] == 1)
            remove_k = int(min(abs(fitness), int(ones_index.size)))
            np.random.shuffle(ones_index)
            to_zero = ones_index[:remove_k]
            X[index, to_zero] = 0
            problem(X[index].tolist()) 
        
    def is_dominating (self, a: np.ndarray, b: np.ndarray, problem: ioh.problem.GraphProblem) -> bool:
        is_no_worse = problem(a.tolist()) > problem(b.tolist()) # fitness

        is_strictly_better = a.sum() <= b.sum() # cost
        
        return is_no_worse and is_strictly_better
    
    
    

    # Example usage inside your search:
    def __call__(self, problem: ioh.problem.GraphProblem):
        n = problem.meta_data.n_variables
        # n = 5

        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # counts = X.sum(axis=1)
        # for i, c in enumerate(counts):
            # print(f"idx={i:2d}  ones={int(c):3d}  x={X[i].tolist()}")

        self._Repair(X, problem)

        
        for i in range(self.pop_size):
            print(f"idx={i:2d}  fitness={problem(X[i].tolist()):6.2f}  ones={int(X[i].sum()):3d}  x={X[i].tolist()}")
        
        for i in range(self.pop_size - 1):
            print(self.is_dominating(X[i], X[i+1], problem))
        
         

