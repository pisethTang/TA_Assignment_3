from .algorithm_interface import Algorithm
import numpy as np
import ioh
from typing import Callable, Optional, Tuple

class MultiObjectiveEA (Algorithm):
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 K_Elites: Optional[int]=None,
                 mutation_prob: Optional[float] = None,
                 name: Optional[str] = "NSGA-II", 
                 algorithm_info: str = "NSGA-II population-based optimisation",
                ):
        super().__init__(budget, name=f"NSGA-II-{int(budget/1000)}K-{pop_size}", algorithm_info=algorithm_info)
        self.rng = np.random.default_rng()  # Added RNG for random operations
        self.pop_size = pop_size
        self.K_Elites = K_Elites
        self.mutation_prob = mutation_prob

    def _Repair(self, X: np.ndarray, fitnesses: np.ndarray, problem: ioh.problem.GraphProblem) -> None:
        for index, _ in enumerate(X):
            if fitnesses[index] >= 0:
                continue

            ones_index = np.flatnonzero(X[index] == 1)
            remove_k = int(min(abs(fitnesses[index]), int(ones_index.size)))
            np.random.shuffle(ones_index)
            to_zero = ones_index[:remove_k]
            X[index, to_zero] = 0
            # Re-eval only this repaired individual
            fitnesses[index] = problem(X[index].tolist())
        
    def _is_dominating (self, a: np.ndarray, fa: float, b: np.ndarray, fb: float) -> bool:
        cost_a = a.sum()
        cost_b = b.sum()
        return fa > fb and cost_a <= cost_b
    
    def _Sort(self, X: np.ndarray, fitnesses: np.ndarray) -> list[list[int]]:
        pop_size = X.shape[0]
        S = [set() for _ in range(pop_size)]
        n = [0] * pop_size
        for p in range(pop_size):
            for q in range(pop_size):
                if p != q:
                    if self._is_dominating(X[p], fitnesses[p], X[q], fitnesses[q]):
                        S[p].add(q)
                    elif self._is_dominating(X[q], fitnesses[q], X[p], fitnesses[p]):
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

    def _CrowdingDist(self, X: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        distances = np.zeros(l)
        
        # Compute objectives: fitness (maximize), cost (minimize)
        costs = X.sum(axis=1)
        
        objs = np.stack((fitnesses, costs), axis=1)  # shape (l, 2)
        
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

    def _EliteNeighbour(self, X: np.ndarray, k: int, fitnesses: np.ndarray) -> np.ndarray:
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

    # Example usage inside your search:
    def __call__(self, problem: ioh.problem.GraphProblem):
        n = problem.meta_data.n_variables

        # Initialize parent population P
        P = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)
        fitnesses = np.array([problem(x.tolist()) for x in P])

        # repair for initial population
        self._Repair(P, fitnesses, problem)

        fronts = self._Sort(P, fitnesses)

        while problem.state.evaluations < self.budget:
            # Create child population Q using classical GA approach: elite selection, UnCX, KMutation
            Elites = self._EliteNeighbour(P, k=self.K_Elites, fitnesses=fitnesses)
            Parents = P[Elites]

            children = []
            for i in range(0, self.pop_size, 2):
                p1_idx = i % len(Parents)
                p2_idx = (i + 1) % len(Parents)
                p1 = Parents[p1_idx]
                p2 = Parents[p2_idx]

                child1, child2 = self._UniformCX(p1, p2)

                children.append(child1)
                children.append(child2)

            Q = np.vstack(children)[:self.pop_size]  # Trim if extra

            self._KMutation(Q)

            # Eval and repair Q
            Q_fitnesses = np.array([problem(x.tolist()) for x in Q])
            self._Repair(Q, Q_fitnesses, problem)

            # Combine R = P U Q
            R = np.vstack([P, Q])
            R_fitnesses = np.concatenate([fitnesses, Q_fitnesses])

            # Non-dominated sort on R
            fronts = self._Sort(R, R_fitnesses)

            # Build P_{t+1}
            new_P = np.zeros((self.pop_size, n), dtype=int)
            new_fitnesses = np.zeros(self.pop_size)
            ptr = 0
            for f in fronts:
                if ptr + len(f) <= self.pop_size:
                    new_P[ptr:ptr + len(f)] = R[f]
                    new_fitnesses[ptr:ptr + len(f)] = R_fitnesses[f]
                    ptr += len(f)
                else:
                    # Crowding for last front
                    front_X = R[f]
                    front_dist = self._CrowdingDist(front_X, R_fitnesses[f])

                    # Sort by descending crowding
                    sort_order = np.argsort(front_dist)[::-1]

                    remaining = self.pop_size - ptr
                    new_P[ptr:ptr + remaining] = front_X[sort_order[:remaining]]
                    new_fitnesses[ptr:ptr + remaining] = R_fitnesses[f][sort_order[:remaining]]
                    break

            # Update P and fitnesses
            P = new_P
            fitnesses = new_fitnesses

            # Recompute fronts, ranks, distances for new P
            fronts = self._Sort(P, fitnesses)