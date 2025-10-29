from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple

class SOP_EA:
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 B = 10,
                 name: str = "SOP_EA", 
                 algorithm_info: str = "SOP population-based optimisation",
                ):
        # Note: Removed super().__init__() since there's no base class specified for bare-bones testing.
        # If this inherits from a base class in the full version, add it back.
        self.budget = budget
        self.pop_size = pop_size
        self.B = B
        self.name = name
        self.algorithm_info = algorithm_info
        self.rng = np.random.default_rng()  # Added RNG for random operations

    #----Cost function--------
    def _cost(self, X: np.ndarray, pop_size: int, dimension: int) -> np.ndarray:
        # Fixed: Costs should be 1D array (one cost per individual)
        expenses = np.zeros(pop_size, dtype=int)

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
    def __call__(self, n: int):  # Modified for bare-bones: Pass dimension directly instead of problem
        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # Get the cost for all the initialisation
        C = self._cost(X, self.pop_size, n)
        self._repair_in_place(X, C)
        # For testing, you could add more logic here (e.g., optimization loop),
        # but keeping it minimal as per the incomplete original.

        return X, C  # Return for testing purposes
    
def main():
    # Example parameters for testing
    budget = 100  # Not used in this bare-bones version, but kept for consistency
    pop_size = 5
    B = 10  # Small B to easily trigger repairs
    n = 20  # Dimension (number of variables)

    # Create instance
    ea = SOP_EA(budget, pop_size, B=B)

    # Call the algorithm (bare-bones, without IOH problem)
    X, C = ea(n)


if __name__ == "__main__":
    main()