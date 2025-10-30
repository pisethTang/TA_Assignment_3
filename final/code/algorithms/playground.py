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
    def _cost(self, X: np.ndarray, pop_size: int) -> np.ndarray:
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
        
    def _EliteOnes(self, X: np.ndarray, k: int) -> np.ndarray:
        # Compute the number of ones for each solution (row)
        ones_counts = X.sum(axis=1).astype(int)
        
        # Get the indices sorted in descending order (highest ones first)
        indices = np.argsort(ones_counts)[::-1]
        
        # Select the top k indices
        top_k_indices = indices[:k]
        
        # Return the indices (you can use X[top_k_indices] to get the actual solutions)
        return top_k_indices

    def _UniformCX(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(p1)
        # Generate a random mask: 1 means take from p1 for c1 (p2 for c2), 0 means take from p2 for c1 (p1 for c2)
        mask = self.rng.integers(0, 2, size=n, dtype=int)
        
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        
        return c1, c2

    def _KMutation(self, X: np.ndarray) -> None:
        # Mutate each individual in X in place
        num_individuals, n = X.shape
        for i in range(num_individuals):
            # Choose random k: 1 <= k <= n//2
            k = self.rng.integers(1, n//2 + 1)
            # Choose k unique random indices
            indices = self.rng.choice(n, size=k, replace=False)
            # Flip the bits at those indices
            X[i, indices] = 1 - X[i, indices]
        
    def _Tournament(self, X: np.ndarray, tournament_size: int, num_to_select: int) -> np.ndarray:
        pop_size, n = X.shape
        ones = X.sum(axis=1)  # Fitness: number of ones (assuming maximization under constraint)
        
        selected_indices = []
        for _ in range(num_to_select):
            # Pick k unique random candidates
            candidates = self.rng.choice(pop_size, tournament_size, replace=False)
            # Select the one with the highest fitness
            best_idx = candidates[np.argmax(ones[candidates])]
            selected_indices.append(best_idx)
        
        return X[np.array(selected_indices)]  # Return the selected individuals

    # Example usage inside your search:
    def __call__(self, n: int):  # Modified for bare-bones: Pass dimension directly instead of problem
        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # Get the cost for all the initialisation
        C = self._cost(X, self.pop_size)
        self._repair_in_place(X, C)
        elites = self._EliteOnes(X,k=3)
        print("Top 3 elites:")
        print(X[elites])

        # Test the crossover with first two elites
        elite_p1 = X[elites[0]]
        elite_p2 = X[elites[1]]
        child1, child2 = self._UniformCX(elite_p1, elite_p2)

        print("\nOffspring before repair:")
        print(child1)
        print(child2)

        print("\nOffspring before mutation and repair:")
        print(child1)
        print(child2)

        # Put them into a temporary population for mutation and repair
        children_X = np.vstack([child1, child2])

        # Apply mutation
        self._KMutation(children_X)

        print("\nOffspring after mutation (before repair):")
        print(children_X[0])
        print(children_X[1])



        # For testing, you could add more logic here (e.g., optimization loop),
        # but keeping it minimal as per the incomplete original.

        return X, C  # Return for testing purposes
    
def main():
    # Example parameters for testing
    budget = 100  # Not used in this bare-bones version, but kept for consistency
    pop_size = 10
    B = 10  # Small B to easily trigger repairs
    n = 20  # Dimension (number of variables)

    # Create instance
    ea = SOP_EA(budget, pop_size, B=B)

    # Call the algorithm (bare-bones, without IOH problem)
    X, C = ea(n)

    print("\nFinal population and costs (after repair):")
    for i in range(pop_size):
        print(f"Individual {i}: {X[i]} (cost: {C[i]})")



if __name__ == "__main__":
    main()