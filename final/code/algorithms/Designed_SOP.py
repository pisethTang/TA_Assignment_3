from .algorithm_interface import Algorithm
import numpy as np
import ioh
from typing import Callable, Optional, Tuple

class SOP_EA(Algorithm):
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 K_Elites: int,
                 mutation_prob: Optional[float] = None,
                 name: str = "SOP_EA", 
                 algorithm_info: str = "SOP population-based optimisation",
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
        
    def _EliteOnes(self, X: np.ndarray, k: int) -> np.ndarray:
        # Compute the number of ones for each solution (row)
        ones_counts = X.sum(axis=1).astype(int)
        
        # Get the indices sorted in descending order (highest ones first)
        indices = np.argsort(ones_counts)[::-1]
        
        # Select the top k indices
        top_k_indices = indices[:k]
        
        # Return the indices (you can use X[top_k_indices] to get the actual solutions)
        return top_k_indices
    
    def _EliteNeighbour(self, X: np.ndarray, k: int, problem: ioh.problem.GraphProblem) -> np.ndarray:
        # Compute the fitness for each solution (row)
        fitnesses = np.array([problem(x.tolist()) for x in X])
        
        # Get the indices sorted in descending order (highest fitness first)
        indices = np.argsort(fitnesses)[::-1]
        
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
            # k = int(n//5)
            # Choose k unique random indices
            indices = self.rng.choice(n, size=k, replace=False)
            # Flip the bits at those indices
            X[i, indices] = 1 - X[i, indices]
        
    def _UniformMutation(self, X: np.ndarray) -> None:
        # Mutate each individual in X in place by flipping each bit with probability mutation_prob (or 1/n if not set)
        num_individuals, n = X.shape
        prob = self.mutation_prob if self.mutation_prob is not None else 1.0 / n
        mask = self.rng.uniform(size=(num_individuals, n)) < prob
        X[mask] = 1 - X[mask]

    def _OnesTournament(self, X: np.ndarray, tournament_size: int, num_to_select: int) -> np.ndarray:
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

    def _NeighbourTournament(self, X: np.ndarray, tournament_size: int, num_to_select: int, problem: ioh.problem.GraphProblem) -> np.ndarray:
        pop_size, n = X.shape
        fitnesses = np.array([problem(x.tolist()) for x in X])  # Fitness: actual problem fitness
        selected_indices = []

        for _ in range(num_to_select):
            # Pick k unique random candidates
            candidates = self.rng.choice(pop_size, tournament_size, replace=False)
            # Select the one with the highest fitness
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected_indices.append(best_idx)
        
        return X[np.array(selected_indices)]  # Return the selected individuals

    # Example usage inside your search:
    def __call__(self, problem: ioh.problem.GraphProblem):
        n = problem.meta_data.n_variables

        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # counts = X.sum(axis=1)
        # for i, c in enumerate(counts):
            # print(f"idx={i:2d}  ones={int(c):3d}  x={X[i].tolist()}")

        self._Repair(X, problem)

        while problem.state.evaluations < self.budget:
            Elites = self._EliteNeighbour(X, k=self.K_Elites,problem=problem)
            Parents = X[Elites].copy()

            k = Parents.shape[0]
            children = []
            for i in range(0,k,2):
                p1 = Parents[i].copy()      # copy to be safe if UniformCX mutates
                p2 = Parents[i+1].copy()

                child1, child2 = self._UniformCX(p1, p2)

                children.append(child1)
                children.append(child2)

            Children = np.vstack(children)   # shape (k, n)
            self._KMutation(Children)

            self._Repair(Children, problem)

            Pool = np.vstack([X, Children])

            next_gen = self._NeighbourTournament(X=Pool, tournament_size=10, num_to_select=self.pop_size,problem=problem)

            # make sure we have the right shape and dtype
            X = np.asarray(next_gen, dtype=int).copy()
