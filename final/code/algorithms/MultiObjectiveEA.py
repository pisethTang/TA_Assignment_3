# from .algorithm_interface import Algorithm
# import numpy as np
# import ioh
# from typing import Callable, Optional, Tuple

# class MultiObjectiveEA (Algorithm):
#     def __init__(self,
#                  budget: int,
#                  population_size: Optional[int] = None,
#                 #  pop_size: Optional[int] = None,
#                  K_Elites: Optional[int] = None,
#                  mutation_prob: Optional[float] = None,
#                  name: Optional[str] = "NSGA-II",
#                  algorithm_info: str = "NSGA-II population-based optimisation",
#                  ):
#         """
#         Parameters
#         - budget: IOH evaluation budget
#         - population_size / pop_size: population size (both names accepted for backwards compatibility)
#         - K_Elites: number of elite parents used to generate children
#         - mutation_prob: per-bit mutation probability (defaults to 1/n when None)
#         """

#         # Accept both keyword names for compatibility with existing code (config, notebooks, etc.)
#         # final_pop_size = population_size if population_size is not None else pop_size
#         # if final_pop_size is None:
#         #     raise TypeError("MultiObjectiveEA requires 'population_size' (or 'pop_size') to be provided")

#         super().__init__(budget, name=f"NSGA-II-{int(budget/1000)}K-{population_size}", algorithm_info=algorithm_info)
#         self.rng = np.random.default_rng()  # RNG for random operations

#         # Keep BOTH attribute names to match other algorithms and printing in main.py
#         self.pop_size = int(population_size)
#         self.population_size = int(population_size)

#         self.K_Elites = K_Elites
#         self.mutation_prob = mutation_prob

#     def _Repair(self, X: np.ndarray, fitnesses: np.ndarray, problem: ioh.problem.GraphProblem) -> None:
#         for index, _ in enumerate(X):
#             if fitnesses[index] >= 0:
#                 continue

#             ones_index = np.flatnonzero(X[index] == 1)
#             remove_k = int(min(abs(fitnesses[index]), int(ones_index.size)))
#             np.random.shuffle(ones_index)
#             to_zero = ones_index[:remove_k]
#             X[index, to_zero] = 0
#             # Re-eval only this repaired individual
#             fitnesses[index] = problem(X[index].tolist())
        
#     def _is_dominating (self, a: np.ndarray, fa: float, b: np.ndarray, fb: float) -> bool:
#         cost_a = a.sum()
#         cost_b = b.sum()
#         return fa > fb and cost_a <= cost_b
    
#     def _Sort(self, X: np.ndarray, fitnesses: np.ndarray) -> list[list[int]]:
#         pop_size = X.shape[0]
#         S = [set() for _ in range(pop_size)]
#         n = [0] * pop_size
#         for p in range(pop_size):
#             for q in range(pop_size):
#                 if p != q:
#                     if self._is_dominating(X[p], fitnesses[p], X[q], fitnesses[q]):
#                         S[p].add(q)
#                     elif self._is_dominating(X[q], fitnesses[q], X[p], fitnesses[p]):
#                         n[p] += 1
#         fronts = []
#         current_front = [p for p in range(pop_size) if n[p] == 0]
#         while current_front:
#             fronts.append(current_front)
#             next_front = []
#             for p in current_front:
#                 for q in S[p]:
#                     n[q] -= 1
#                     if n[q] == 0:
#                         next_front.append(q)
#             current_front = next_front
#         return fronts

#     def _CrowdingDist(self, X: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
#         l = X.shape[0]
#         distances = np.zeros(l)
        
#         # Compute objectives: fitness (maximize), cost (minimize)
#         costs = X.sum(axis=1)
        
#         objs = np.stack((fitnesses, costs), axis=1)  # shape (l, 2)
        
#         for m in range(2):
#             # Sort indices ascending by objective m
#             idx = np.argsort(objs[:, m])
            
#             # Boundary points get infinity
#             distances[idx[0]] = np.inf
#             distances[idx[-1]] = np.inf
            
#             # Avoid divide by zero if all values equal
#             fmin = objs[idx[0], m]
#             fmax = objs[idx[-1], m]
#             if fmax == fmin:
#                 continue
            
#             # For middle points
#             for i in range(1, l - 1):
#                 diff = objs[idx[i + 1], m] - objs[idx[i - 1], m]
#                 distances[idx[i]] += diff / (fmax - fmin)
        
#         return distances
    
#     def _CrowdedComparison(self, i: int, j: int, ranks: np.ndarray, distances: np.ndarray) -> bool:
#         # Returns True if i <_n j (i is better than j)
#         if ranks[i] < ranks[j]:
#             return True
#         if ranks[i] > ranks[j]:
#             return False
#         # Same rank
#         return distances[i] > distances[j]

#     def _EliteNeighbour(self, X: np.ndarray, k: int, fitnesses: np.ndarray) -> np.ndarray:
#         # Get the indices sorted in descending order (highest fitness first)
#         indices = np.argsort(fitnesses)[::-1]
        
#         # Select the top k indices
#         top_k_indices = indices[:k]
        
#         # Return the indices (you can use X[top_k_indices] to get the actual solutions)
#         return top_k_indices

#     def _UniformCX(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         n = len(p1)
#         # Generate a random mask: 1 means take from p1 for c1 (p2 for c2), 0 means take from p2 for c1 (p1 for c2)
#         mask = self.rng.integers(0, 2, size=n, dtype=int)
        
#         c1 = np.where(mask, p1, p2)
#         c2 = np.where(mask, p2, p1)
        
#         return c1, c2

#     def _KMutation(self, X: np.ndarray) -> None:
#         # Mutate each individual in X in place
#         num_individuals, n = X.shape
#         for i in range(num_individuals):
#             # Choose random k: 1 <= k <= n//2
#             k = self.rng.integers(1, n//2 + 1)
#             # k = int(n//5)
#             # Choose k unique random indices
#             indices = self.rng.choice(n, size=k, replace=False)
#             # Flip the bits at those indices
#             X[i, indices] = 1 - X[i, indices]
        
#     def _UniformMutation(self, X: np.ndarray) -> None:
#         # Mutate each individual in X in place by flipping each bit with probability mutation_prob (or 1/n if not set)
#         num_individuals, n = X.shape
#         prob = self.mutation_prob if self.mutation_prob is not None else 1.0 / n
#         mask = self.rng.uniform(size=(num_individuals, n)) < prob
#         X[mask] = 1 - X[mask]

#     def _OnesTournament(self, X: np.ndarray, tournament_size: int, num_to_select: int) -> np.ndarray:
#         pop_size, n = X.shape
#         ones = X.sum(axis=1)  # Fitness: number of ones (assuming maximization under constraint)
        
#         selected_indices = []
#         for _ in range(num_to_select):
#             # Pick k unique random candidates
#             candidates = self.rng.choice(pop_size, tournament_size, replace=False)
#             # Select the one with the highest fitness
#             best_idx = candidates[np.argmax(ones[candidates])]
#             selected_indices.append(best_idx)
        
#         return X[np.array(selected_indices)]  # Return the selected individuals

#     # Example usage inside your search:
#     def __call__(self, problem: ioh.problem.GraphProblem):
#         n = problem.meta_data.n_variables

#         # Initialize parent population P
#         P = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=int)
#         fitnesses = np.array([problem(x.tolist()) for x in P])

#         # repair for initial population
#         self._Repair(P, fitnesses, problem)

#         fronts = self._Sort(P, fitnesses)

#         while problem.state.evaluations < self.budget:
#             # Create child population Q using classical GA approach: elite selection, UnCX, KMutation
#             Elites = self._EliteNeighbour(P, k=self.K_Elites, fitnesses=fitnesses)
#             Parents = P[Elites]

#             children = []
#             for i in range(0, self.pop_size, 2):
#                 p1_idx = i % len(Parents)
#                 p2_idx = (i + 1) % len(Parents)
#                 p1 = Parents[p1_idx]
#                 p2 = Parents[p2_idx]

#                 child1, child2 = self._UniformCX(p1, p2)

#                 children.append(child1)
#                 children.append(child2)

#             Q = np.vstack(children)[:self.pop_size]  # Trim if extra

#             self._KMutation(Q)

#             # Eval and repair Q
#             Q_fitnesses = np.array([problem(x.tolist()) for x in Q])
#             self._Repair(Q, Q_fitnesses, problem)

#             # Combine R = P U Q
#             R = np.vstack([P, Q])
#             R_fitnesses = np.concatenate([fitnesses, Q_fitnesses])

#             # Non-dominated sort on R
#             fronts = self._Sort(R, R_fitnesses)

#             # Build P_{t+1}
#             new_P = np.zeros((self.pop_size, n), dtype=int)
#             new_fitnesses = np.zeros(self.pop_size)
#             ptr = 0
#             for f in fronts:
#                 if ptr + len(f) <= self.pop_size:
#                     new_P[ptr:ptr + len(f)] = R[f]
#                     new_fitnesses[ptr:ptr + len(f)] = R_fitnesses[f]
#                     ptr += len(f)
#                 else:
#                     # Crowding for last front
#                     front_X = R[f]
#                     front_dist = self._CrowdingDist(front_X, R_fitnesses[f])

#                     # Sort by descending crowding
#                     sort_order = np.argsort(front_dist)[::-1]

#                     remaining = self.pop_size - ptr
#                     new_P[ptr:ptr + remaining] = front_X[sort_order[:remaining]]
#                     new_fitnesses[ptr:ptr + remaining] = R_fitnesses[f][sort_order[:remaining]]
#                     break

#             # Update P and fitnesses
#             P = new_P
#             fitnesses = new_fitnesses

#             # Recompute fronts, ranks, distances for new P
#             fronts = self._Sort(P, fitnesses)

#             # break if optimum is found 
#             if problem.state.optimum_found:
#                 break
import numpy as np
import ioh
from typing import Optional, Tuple, Dict, List

class MultiObjectiveEA:
    def __init__(self,
                 budget: int,
                 population_size: int,
                 K_Elites: Optional[int] = None,
                 mutation_prob: Optional[float] = None,  # used if mutation_mode='uniform'
                 mutation_mode: str = "hybrid",           # 'uniform' | 'heavy' | 'hybrid'
                 beta: float = 1.5,                      # heavy-tailed exponent
                 heavy_mix: float = 0.15,                # fraction of children using heavy mutation in 'hybrid'
                 k_max: int = 10,                        # cap flips in heavy mutation
                 B: int = 10,                            # known budget (max ones)
                 parent_selection: str = "crowded",      # 'crowded' | 'elites'
                 name: Optional[str] = None,
                 algorithm_info: str = "NSGA-II (cached, pre-repair, hybrid mutation)"):
        """
        budget: IOH evaluation budget
        population_size: population size
        K_Elites: used only when parent_selection='elites'
        mutation_prob: per-bit probability for 'uniform'/'hybrid' (default 1/n if None)
        mutation_mode: 'uniform', 'heavy', or 'hybrid'
        beta: heavy-tailed exponent (1.5 recommended)
        heavy_mix: fraction of children using heavy mutation when mutation_mode='hybrid'
        k_max: max flips for heavy mutation (keeps feasibility under control)
        B: cost constraint (max number of ones); pre-clamp before evaluation
        parent_selection: 'crowded' (recommended) or 'elites'
        """
        self.budget = int(budget)
        self.pop_size = int(population_size)
        self.population_size = int(population_size)
        self.K_Elites = int(K_Elites) if K_Elites is not None else max(2, self.pop_size // 2)
        self.mutation_prob = mutation_prob
        self.mutation_mode = mutation_mode
        self.beta = float(beta)
        self.heavy_mix = float(heavy_mix)
        self.k_max = int(k_max)
        self.B = int(B)
        self.parent_selection = parent_selection
        self.name = name or f"NSGA-II-{int(self.budget/1000)}K-{self.pop_size}"
        self.algorithm_info = algorithm_info

        self.rng = np.random.default_rng()
        self._eval_cache: Dict[bytes, float] = {}

    # ---------- caching / eval ----------
    def _key(self, x: np.ndarray) -> bytes:
        return np.packbits(x, bitorder='little').tobytes()

    def _eval_one(self, x: np.ndarray, problem: ioh.problem.GraphProblem) -> float:
        k = self._key(x)
        if k in self._eval_cache:
            return self._eval_cache[k]
        f = problem(x.tolist())
        self._eval_cache[k] = f
        return f

    def _eval_batch(self, X: np.ndarray, problem: ioh.problem.GraphProblem) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            out[i] = self._eval_one(X[i], problem)
        return out

    # ---------- dominance / sorting ----------
    def _dominates(self, a: np.ndarray, fa: float, b: np.ndarray, fb: float) -> bool:
        # objectives: maximize fitness, minimize cost (ones)
        ca, cb = a.sum(), b.sum()
        return (fa >= fb and ca <= cb) and (fa > fb or ca < cb)

    def _fast_non_dominated_sort(self, X: np.ndarray, F: np.ndarray) -> List[List[int]]:
        nP = X.shape[0]
        S = [set() for _ in range(nP)]
        n = [0] * nP
        for p in range(nP):
            for q in range(p + 1, nP):
                pdq = self._dominates(X[p], F[p], X[q], F[q])
                qdp = self._dominates(X[q], F[q], X[p], F[p])
                if pdq and not qdp:
                    S[p].add(q)
                    n[q] += 1
                elif qdp and not pdq:
                    S[q].add(p)
                    n[p] += 1
        fronts = []
        current = [i for i in range(nP) if n[i] == 0]
        while current:
            fronts.append(current)
            nxt = []
            for p in current:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        nxt.append(q)
            current = nxt
        return fronts

    def _crowding_distance(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        if l == 0:
            return np.array([], dtype=float)
        d = np.zeros(l, dtype=float)
        cost = X.sum(axis=1)
        objs = np.stack((F, -cost), axis=1)  # larger is better for both columns
        for m in range(2):
            idx = np.argsort(objs[:, m])  # ascending
            d[idx[0]] = np.inf
            d[idx[-1]] = np.inf
            fmin, fmax = objs[idx[0], m], objs[idx[-1], m]
            if fmax == fmin or l <= 2:
                continue
            numer = objs[idx[2:], m] - objs[idx[:-2], m]
            d[idx[1:-1]] += numer / (fmax - fmin)
        return d

    # ---------- parent selection ----------
    def _select_parents(self, P: np.ndarray, F: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
        if self.parent_selection == "elites":
            elite_idx = np.argsort(F)[::-1][:self.K_Elites]
            # round-robin from elites
            parents = np.array([elite_idx[i % elite_idx.size] for i in range(self.pop_size)], dtype=int)
            return parents

        # crowded binary tournament (recommended)
        nP = P.shape[0]
        ranks = np.full(nP, fill_value=10**9, dtype=int)
        distances = np.zeros(nP, dtype=float)
        for r, f in enumerate(fronts):
            ranks[f] = r
            distances[f] = self._crowding_distance(P[f], F[f])

        def better(i, j) -> bool:
            if ranks[i] != ranks[j]:
                return ranks[i] < ranks[j]
            return distances[i] > distances[j]

        parents = np.empty(self.pop_size, dtype=int)
        for t in range(self.pop_size):
            a, b = self.rng.integers(0, nP, size=2)
            parents[t] = a if better(a, b) else b
        return parents

    # ---------- variation ----------
    def _uniform_cx(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = p1.size
        mask = self.rng.integers(0, 2, size=n, dtype=np.int8)
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    def _mutate_uniform(self, X: np.ndarray) -> None:
        num, n = X.shape
        p = self.mutation_prob if self.mutation_prob is not None else (1.0 / n)
        mask = self.rng.random((num, n)) < p
        X[mask] ^= 1

    def _mutate_heavy(self, X: np.ndarray) -> None:
        num, n = X.shape
        maxk = min(self.k_max, max(1, n // 2))
        for i in range(num):
            k = maxk + 1
            while k > maxk:
                k = int(self.rng.zipf(self.beta))
            idx = self.rng.choice(n, size=k, replace=False)
            X[i, idx] ^= 1

    def _mutate_hybrid(self, X: np.ndarray) -> None:
        num = X.shape[0]
        if num == 0:
            return
        m_heavy = max(1, int(round(self.heavy_mix * num)))
        idx = self.rng.permutation(num)
        heavy_idx, light_idx = idx[:m_heavy], idx[m_heavy:]
        if heavy_idx.size:
            self._mutate_heavy(X[heavy_idx])
        if light_idx.size:
            # use uniform p=1/n (or provided)
            sub = X[light_idx]
            self._mutate_uniform(sub)
            X[light_idx] = sub

    def _mutate(self, X: np.ndarray) -> None:
        if self.mutation_mode == "uniform":
            self._mutate_uniform(X)
        elif self.mutation_mode == "heavy":
            self._mutate_heavy(X)
        else:
            self._mutate_hybrid(X)

    # ---------- feasibility helpers ----------
    def _pre_clamp_to_B(self, X: np.ndarray) -> None:
        # enforce ones <= B BEFORE eval (avoids double eval + repair)
        ones = X.sum(axis=1)
        viol = np.where(ones > self.B)[0]
        for i in viol:
            pos = np.flatnonzero(X[i] == 1)
            self.rng.shuffle(pos)
            # Ensure Python int for slicing
            drop = int(ones[i] - self.B)
            if drop > 0:
                X[i, pos[:drop]] = 0

    def _repair_if_needed(self, X: np.ndarray, F: np.ndarray, problem: ioh.problem.GraphProblem) -> None:
        # If IOH signals infeasible by negative fitness, try up to 2 trims
        for i in range(X.shape[0]):
            if F[i] >= 0:
                continue
            # aggressive trim towards B
            pos = np.flatnonzero(X[i] == 1)
            if pos.size == 0:
                continue
            self.rng.shuffle(pos)
            # Remove enough to be at most B (fallback remove 20% if unknown)
            cur = pos.size
            to_remove = max(cur - self.B, max(1, cur // 5))
            X[i, pos[:to_remove]] = 0
            F[i] = self._eval_one(X[i], problem)
            if F[i] < 0:
                # second attempt
                pos = np.flatnonzero(X[i] == 1)
                if pos.size:
                    self.rng.shuffle(pos)
                    X[i, pos[:max(1, pos.size // 5)]] = 0
                    F[i] = self._eval_one(X[i], problem)

    # ---------- main ----------
    def __call__(self, problem: ioh.problem.GraphProblem):
        n = problem.meta_data.n_variables
        self._eval_cache.clear()

        # init P
        P = self.rng.integers(0, 2, size=(self.pop_size, n), dtype=np.uint8)
        self._pre_clamp_to_B(P)
        F = self._eval_batch(P, problem)
        self._repair_if_needed(P, F, problem)

        fronts = self._fast_non_dominated_sort(P, F)

        while problem.state.evaluations < self.budget:
            # parent selection
            parent_idx = self._select_parents(P, F, fronts)

            # crossover
            children = np.empty_like(P)
            for i in range(0, self.pop_size, 2):
                p1 = P[parent_idx[i]]
                p2 = P[parent_idx[(i + 1) % self.pop_size]]
                c1, c2 = self._uniform_cx(p1, p2)
                children[i] = c1
                if i + 1 < self.pop_size:
                    children[i + 1] = c2

            # mutate
            self._mutate(children)

            # pre-clamp then eval + repair
            self._pre_clamp_to_B(children)
            Q = children
            QF = self._eval_batch(Q, problem)
            self._repair_if_needed(Q, QF, problem)

            # combine and select next P
            R = np.vstack([P, Q])
            RF = np.concatenate([F, QF])
            fronts = self._fast_non_dominated_sort(R, RF)

            new_P = np.empty_like(P)
            new_F = np.empty_like(F)
            ptr = 0
            for fr in fronts:
                fr = np.array(fr, dtype=int)
                if ptr + fr.size <= self.pop_size:
                    new_P[ptr:ptr + fr.size] = R[fr]
                    new_F[ptr:ptr + fr.size] = RF[fr]
                    ptr += fr.size
                else:
                    dist = self._crowding_distance(R[fr], RF[fr])
                    order = np.argsort(dist)[::-1]
                    k = self.pop_size - ptr
                    choose = fr[order[:k]]
                    new_P[ptr:ptr + k] = R[choose]
                    new_F[ptr:ptr + k] = RF[choose]
                    ptr += k
                    break

            P, F = new_P, new_F
            fronts = self._fast_non_dominated_sort(P, F)

            if problem.state.optimum_found:
                break