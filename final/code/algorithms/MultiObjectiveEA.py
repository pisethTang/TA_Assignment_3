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
        - `budget`: IOH evaluation budget
        - `population_size`: population size
        - `K_Elites`: used only when parent_selection='elites'
        - `mutation_prob`: per-bit probability for 'uniform'/'hybrid' (default 1/n if None)
        - `mutation_mode`: 'uniform', 'heavy', or 'hybrid'
        - `beta`: heavy-tailed exponent (1.5 recommended)
        - `heavy_mix`: fraction of children using heavy mutation when mutation_mode='hybrid'
        - `k_max`: max flips for heavy mutation (keeps feasibility under control)
        - `B`: cost constraint (max number of ones); pre-clamp before evaluation
        - `parent_selection`: 'crowded' (recommended) or 'elites'
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