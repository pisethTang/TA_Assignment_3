from __future__ import annotations
from .algorithm_interface import Algorithm
import numpy as np
import ioh
from typing import Callable, Optional, Tuple

class SOP_EA(Algorithm):
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 B = 10,
                 name: str = "SOP_EA", 
                 algorithm_info: str = "SOP population-based optimisation",
                ):
        super.__init__(budget, name, algorithm_info)
        self.pop_size = pop_size
        self.B = B


    #----Cost function--------
    def _cost(self, X: np.ndarray, pop_size: int, dimension: int) -> np.ndarray:
        expenses = np.zeros((pop_size, dimension), dtype=int)

        for index, x in enumerate(X):
            expenses[index] = int(sum(x))

        return expenses

    def _repair_in_place(self, X: np.ndarray, Cost: np.ndarray) -> None:
        for index, c in enumerate(Cost):
            violation = c - self.B
            if violation <= 0:
                continue

            ones_index = np.flatnonzero(X[index] == 1)
            remove_k = min(violation, ones_index.size)
            np.random.shuffle(ones_index)
            to_zero = ones_index[:remove_k]
            X[index, to_zero] = 0
            Cost[index] = int(X[index].sum())
        

                

    # Example usage inside your search:
    def __call__(self, problem: ioh.Problem):
        n = problem.meta_data.n_variables

        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # Get the cost for all the initialisation
        C = self._cost(X, self.pop_size, n)


        return
