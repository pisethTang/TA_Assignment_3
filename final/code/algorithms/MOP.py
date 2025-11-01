from .algorithm_interface import Algorithm
import numpy as np
import ioh
from typing import Callable, Optional, Tuple

class MOP (Algorithm):
    def __init__(self, 
                 budget: int, 
                 pop_size: int, 
                 K_Elites: Optional[int]=None,
                 mutation_prob: Optional[float] = None,
                 name: Optional[str] = "MOP", 
                 algorithm_info: str = "MOP population-based optimisation",
                ):
        super().__init__(budget, name=f"MOP-{int(budget/1000)}K-{pop_size}", algorithm_info=algorithm_info)
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
        fa = problem(a.tolist())
        fb = problem(b.tolist())
        cost_a = a.sum()
        cost_b = b.sum()
        return (fa >= fb and cost_a <= cost_b) and (fa > fb or cost_a < cost_b)
    
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

        # repair for initial population
        self._Repair(P, problem)

        # Compute initial fronts, ranks, distances for P
        fronts = self._Sort(P, problem)
        ranks = np.zeros(self.pop_size, dtype=int)
        for r, f in enumerate(fronts):
            for idx in f:
                ranks[idx] = r
        distances = np.zeros(self.pop_size)
        for f in fronts:
            if len(f) <= 2:
                distances[f] = np.inf
                continue
            front_X = P[f]
            front_dist = self._CrowdingDist(front_X, problem)
            distances[f] = front_dist

        while problem.state.evaluations < self.budget:
            # Create child population Q
            Q = np.zeros((self.pop_size, n), dtype=int)

            for pair in range(self.pop_size // 2):
                # Binary tournament for parent1
                cand1 = self.rng.integers(self.pop_size)
                cand2 = self.rng.integers(self.pop_size)
                p1_idx = cand1 if self._CrowdedComparison(cand1, cand2, ranks, distances) else cand2

                # Binary tournament for parent2
                cand1 = self.rng.integers(self.pop_size)
                cand2 = self.rng.integers(self.pop_size)
                p2_idx = cand1 if self._CrowdedComparison(cand1, cand2, ranks, distances) else cand2

                # Crossover
                child1, child2 = self._UniformCX(P[p1_idx], P[p2_idx])

                # Add to Q
                Q[pair*2] = child1
                Q[pair*2 + 1] = child2

            # If pop_size odd, add one more child by copying a selected parent
            if self.pop_size % 2 == 1:
                cand1 = self.rng.integers(self.pop_size)
                cand2 = self.rng.integers(self.pop_size)
                p_idx = cand1 if self._CrowdedComparison(cand1, cand2, ranks, distances) else cand2
                Q[-1] = P[p_idx].copy()

            # Mutation
            self._KMutation(Q)

            # Repair
            self._Repair(Q, problem)

            # Combine R = P U Q
            R = np.concatenate((P, Q), axis=0)

            # Non-dominated sort on R
            fronts = self._Sort(R, problem)

            # Build new P
            new_P = np.zeros((self.pop_size, n), dtype=int)
            ptr = 0
            for f in fronts:
                if ptr + len(f) <= self.pop_size:
                    new_P[ptr:ptr + len(f)] = R[f]
                    ptr += len(f)
                else:
                    # Crowding distance for this front
                    front_X = R[f]
                    front_dist = self._CrowdingDist(front_X, problem)

                    # Sort front by descending crowding distance
                    sort_order = np.argsort(front_dist)[::-1]

                    remaining = self.pop_size - ptr
                    new_P[ptr:ptr + remaining] = front_X[sort_order[:remaining]]
                    break

            # Update P
            P = new_P

            # Recompute fronts, ranks, distances for new P
            fronts = self._Sort(P, problem)
            ranks = np.zeros(self.pop_size, dtype=int)
            for r, f in enumerate(fronts):
                for idx in f:
                    ranks[idx] = r
            distances = np.zeros(self.pop_size)
            for f in fronts:
                if len(f) <= 2:
                    distances[f] = np.inf
                    continue
                front_X = P[f]
                front_dist = self._CrowdingDist(front_X, problem)
                distances[f] = front_dist