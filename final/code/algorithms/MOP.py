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
        
    def _is_dominating (self, a: np.ndarray, b: np.ndarray, problem: ioh.problem.GraphProblem) -> bool:
        is_no_worse = problem(a.tolist()) > problem(b.tolist()) # fitness

        is_strictly_better = a.sum() <= b.sum() # cost
        
        return is_no_worse and is_strictly_better
    

    def _Sort(self, X: np.ndarray, problem: ioh.problem.GraphProblem) -> list[list[int]]:
        pop_size = X.shape[0]
        S = [set() for _ in range(pop_size)]
        n = [0] * pop_size
        for p in range(pop_size):
            for q in range(pop_size):
                if p != q:
                    if self._is_dominating(X[p], X[q], problem):
                        S[p].add(q)
                    elif self._is_dominating(X[q], X[p], problem):
                        n[p] += 1
        fronts = []
        current_front = [p for p in range(pop_size) if n[p] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            current_front = next_front
        return fronts

    def _CrowdingDist(self, X: np.ndarray, problem: ioh.problem.GraphProblem) -> np.ndarray:
        l = X.shape[0]
        distances = np.zeros(l)
        
        # Compute objectives: fitness (maximize), cost (minimize)
        fitness = np.array([problem(x.tolist()) for x in X])
        costs = X.sum(axis=1)
        
        objs = np.stack((fitness, costs), axis=1)  # shape (l, 2)
        
        for m in range(2):
            # Sort indices ascending by objective m
            idx = np.argsort(objs[:, m])
            
            # Boundary points get infinity
            distances[idx[0]] = np.inf
            distances[idx[-1]] = np.inf
            
            # Avoid divide by zero if all values equal
            fmin = objs[idx[0], m]
            fmax = objs[idx[-1], m]
            if fmax == fmin:
                continue
            
            # For middle points
            for i in range(1, l - 1):
                diff = objs[idx[i + 1], m] - objs[idx[i - 1], m]
                distances[idx[i]] += diff / (fmax - fmin)
        
        return distances
    
    
    def _CrowdedComparison(self, i: int, j: int, ranks: np.ndarray, distances: np.ndarray) -> bool:
        # Returns True if i <_n j (i is better than j)
        if ranks[i] < ranks[j]:
            return True
        if ranks[i] > ranks[j]:
            return False
        # Same rank
        return distances[i] > distances[j]
    
    # Example usage inside your search:
    def __call__(self, problem: ioh.problem.GraphProblem):
        n = problem.meta_data.n_variables
        # n = 5

        X = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)

        # counts = X.sum(axis=1)
        # for i, c in enumerate(counts):
            # print(f"idx={i:2d}  ones={int(c):3d}  x={X[i].tolist()}")

        self._Repair(X, problem)

        
        # for i in range(self.pop_size):
        #     print(f"idx={i:2d}  fitness={problem(X[i].tolist()):6.2f}  ones={int(X[i].sum()):3d}  x={X[i].tolist()}")
        
        # for i in range(self.pop_size - 1):
        #     print(self._is_dominating(X[i], X[i+1], problem))

        fronts = self._Sort(X, problem)

        # Print results for verification
        print("Test Population:")
        for i, x in enumerate(X):
            fitness = problem(x.tolist())
            cost = x.sum()
            print(f"Idx {i}: x={x.tolist()}, fitness={fitness}, cost={cost}")

        print("\nNon-Dominated Fronts:")
        for rank, front in enumerate(fronts, start=1):
            print(f"Front {rank}: Indices {front}")
            for idx in front:
                x = X[idx]
                fitness = problem(x.tolist())
                cost = x.sum()
                print(f"  - Idx {idx}: fitness={fitness}, cost={cost}")
            
        
        if fronts:
            first_front_indices = fronts[4]
            front_X = X[first_front_indices]
            crowding_distances = self._CrowdingDist(front_X, problem)
            
            print("\nCrowding Distances for First Front:")
            for i, idx in enumerate(first_front_indices):
                dist = crowding_distances[i]
                x = X[idx]
                fitness = problem(x.tolist())
                cost = x.sum()
                print(f"  - Idx {idx}: fitness={fitness}, cost={cost}, crowding_dist={dist:.2f}")
                

