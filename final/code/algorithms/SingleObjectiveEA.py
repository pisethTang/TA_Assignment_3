# from .algorithm_interface import Algorithm
# import ioh 
# import numpy as np



# class SingleObjectiveEA(Algorithm):
#     '''
#     Optimized population-based Fast GA using Power Law Mutation for monotone submodular graph problems.
#     Fast version with minimal repairs and early stopping.
#     '''
#     def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
#                 tournament_size: int = 3, crossover_rate: float = 0.7):
#         super().__init__(budget, name="Fast-GA", 
#                         algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta}, cx={crossover_rate})")
#         self.population_size = population_size
#         self.budget = budget 
#         self.beta = beta
#         self.tournament_size = tournament_size
#         self.crossover_rate = crossover_rate
#         self.power_law_distribution = None

#     # ==================== POWER-LAW MUTATION ====================
    
#     def compute_power_law_distribution(self, n: int) -> np.ndarray:
#         '''Compute power-law distribution (cached).'''
#         distribution = np.zeros(n + 1)
        
#         C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
#         for i in range(1, min(n // 2 + 1, n + 1)):
#             distribution[i] = (i ** (-self.beta)) / C
        
#         distribution[0] = 0.0
#         return distribution

#     def mutate(self, individual: np.ndarray, n: int) -> np.ndarray:
#         '''Fast power-law mutation.'''
#         offspring = individual.copy()
        
#         if self.power_law_distribution is None:
#             self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         k = np.random.choice(n + 1, p=self.power_law_distribution)
#         k = max(1, k)  # At least 1 flip
        
#         flip_indices = np.random.choice(n, size=k, replace=False)
#         offspring[flip_indices] = 1 - offspring[flip_indices]
        
#         return offspring

#     # ==================== CROSSOVER ====================
    
#     def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
#         '''
#         Uniform crossover: each bit chosen randomly from either parent.
#         Better for graph problems than one-point crossover.
#         '''
#         n = len(parent1)
#         offspring = np.where(np.random.rand(n) < 0.5, parent1, parent2)
#         return offspring

#     # ==================== FAST REPAIR (MINIMAL) ====================
    
#     def quick_repair(self, individual: np.ndarray, func) -> tuple:
#         '''
#         ONE-SHOT repair: remove nodes once, don't iterate.
#         '''
#         solution = individual.copy()
#         fitness = func(solution.tolist())
        
#         if fitness >= 0:
#             return solution, fitness
        
#         # ONE-TIME removal: remove 20% of selected nodes
#         ones_positions = np.where(solution == 1)[0]
        
#         if len(ones_positions) > 0:
#             num_to_remove = max(1, len(ones_positions) // 5)
#             remove_positions = np.random.choice(ones_positions, size=num_to_remove, replace=False)
#             solution[remove_positions] = 0
            
#             # Re-evaluate ONCE
#             if func.state.evaluations < self.budget:
#                 fitness = func(solution.tolist())
        
#         return solution, fitness

#     # ==================== FAST SELECTION ====================
    
#     def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
#         '''
#         Return INDEX only (no copying). Fully vectorized.
#         '''
#         pop_size = len(population)
#         tournament_idx = np.random.choice(
#             pop_size, 
#             size=min(self.tournament_size, pop_size), 
#             replace=False
#         )
        
#         # Vectorized: extract fitness values for tournament
#         tournament_fits = fitnesses[tournament_idx]
#         feasible_mask = tournament_fits >= 0
        
#         if np.any(feasible_mask):
#             # Among feasible: pick best fitness
#             feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
#             best_local = np.argmax(feasible_fits)
#         else:
#             # All infeasible: pick least infeasible
#             best_local = np.argmax(tournament_fits)
        
#         return tournament_idx[best_local]

#     # ==================== MAIN ALGORITHM ====================
    
#     def __call__(self, func: ioh.problem.PBO):
#         n = func.meta_data.n_variables
        
#         # Pre-compute power-law distribution
#         self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         # Initialize with reasonable density (10-20% selected)
#         population = []
#         for _ in range(self.population_size):
#             individual = np.zeros(n, dtype=int)
#             # Start with 10-20% of nodes selected (better for graph problems)
#             num_ones = np.random.randint(max(1, n // 10), max(2, n // 5))
#             ones_positions = np.random.choice(n, size=num_ones, replace=False)
#             individual[ones_positions] = 1
#             population.append(individual)
        
#         # Quick initial repair (one-shot, not iterative)
#         fitnesses = np.zeros(self.population_size)
#         for i in range(self.population_size):
#             if func.state.evaluations >= self.budget:
#                 fitnesses = fitnesses[:i]  # Truncate if budget ran out
#                 population = population[:i]
#                 break
            
#             population[i], fitnesses[i] = self.quick_repair(population[i], func)

#         # Early stopping (disabled - let it run the full budget)
#         best_fitness = np.max(fitnesses) if len(fitnesses) > 0 else -np.inf
#         gens_no_improvement:int = 0
        


#         # max_gens_no_improvement:int = float('inf')  # Never stop early
#         # max_gens_no_improvement = max(100, self.budget // (self.population_size * 2))
#         generation = 0
        
#         # Main evolutionary loop
#         while func.state.evaluations < self.budget:
#             if func.state.optimum_found:
#                 break

#             # Early stopping check
#             # if gens_no_improvement >= max_gens_no_improvement:
#             #     break
            
#             # Track feasibility for adaptive repair (vectorized)
#             # num_feasible = np.sum(fitnesses >= 0)

#             # Generate offspring
#             offspring_population = []
#             offspring_fitnesses = []
            
#             for _ in range(self.population_size):
#                 if func.state.evaluations >= self.budget:
#                     break

#                 # Select TWO parents for crossover
#                 parent1_idx = self.tournament_select_fast(population, fitnesses)
#                 parent2_idx = self.tournament_select_fast(population, fitnesses)
                
#                 parent1 = population[parent1_idx]
#                 parent2 = population[parent2_idx]
                
#                 # Crossover (with probability crossover_rate)
#                 if np.random.rand() < self.crossover_rate:
#                     offspring = self.uniform_crossover(parent1, parent2)
#                 else:
#                     # No crossover: just copy one parent
#                     offspring = parent1.copy()
                
#                 # Mutate
#                 offspring = self.mutate(offspring, n)
                
#                 # Evaluate
#                 offspring_fitness = func(offspring.tolist())

#                 # Repair infeasible solutions more aggressively
#                 if offspring_fitness < 0:
#                     # Repair if we have budget left
#                     if func.state.evaluations < self.budget - 10:
#                         offspring, offspring_fitness = self.quick_repair(offspring, func)
                
#                 offspring_population.append(offspring)
#                 offspring_fitnesses.append(offspring_fitness)
            
#             if not offspring_population:
#                 break
            
#             # Fast survival selection with NumPy
#             combined_pop = population + offspring_population


#             offspring_fitnesses = np.array(offspring_fitnesses)
#             combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
#             # Vectorized feasibility check
#             feasible_mask = (combined_fitnesses >= 0).astype(int)
            
#             # lexsort sorts by keys from LAST to FIRST (bottom to top)
#             # So we pass: (fitness, feasibility) to sort by feasibility first, then fitness
#             # Negate values to get descending order
#             sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
#             # Keep top population_size
#             top_indices = sorted_indices[:self.population_size]
#             population = [combined_pop[i] for i in top_indices]
#             fitnesses = combined_fitnesses[top_indices] 
            
#             # Track improvement for early stopping (vectorized)
#             current_best = np.max(fitnesses)
#             if current_best > best_fitness: # Improvement
#                 best_fitness = current_best
#                 gens_no_improvement = 0    
#             else:                           # No improvement
#                 gens_no_improvement += 1
            
#             generation += 1







# from .algorithm_interface import Algorithm
# import ioh 
# import numpy as np


# class SingleObjectiveEA(Algorithm):
#     '''
#     Optimized population-based Fast GA using Power Law Mutation for monotone submodular graph problems.
#     Fast version with adaptive initialization, smart repairs, and balanced crossover.
#     '''
#     def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
#                 tournament_size: int = 3, crossover_rate: float = 0.5):
#         super().__init__(budget, name="Fast-GA", 
#                         algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta}, cx={crossover_rate})")
#         self.population_size = population_size
#         self.budget = budget 
#         self.beta = beta
#         self.tournament_size = tournament_size
#         self.crossover_rate = crossover_rate  # Reduced from 0.7 to 0.5
#         self.power_law_distribution = None

#     # ==================== POWER-LAW MUTATION ====================
    
#     def compute_power_law_distribution(self, n: int) -> np.ndarray:
#         '''Compute power-law distribution (cached).'''
#         distribution = np.zeros(n + 1)
        
#         C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
#         for i in range(1, min(n // 2 + 1, n + 1)):
#             distribution[i] = (i ** (-self.beta)) / C
        
#         distribution[0] = 0.0
#         return distribution

#     def mutate(self, individual: np.ndarray, n: int) -> np.ndarray:
#         '''Fast power-law mutation.'''
#         offspring = individual.copy()
        
#         if self.power_law_distribution is None:
#             self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         k = np.random.choice(n + 1, p=self.power_law_distribution)
#         k = max(1, k)  # At least 1 flip
        
#         flip_indices = np.random.choice(n, size=k, replace=False)
#         offspring[flip_indices] = 1 - offspring[flip_indices]
        
#         return offspring

#     # ==================== CROSSOVER ====================
    
#     def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
#         '''Uniform crossover: each bit chosen randomly from either parent.'''
#         n = len(parent1)
#         offspring = np.where(np.random.rand(n) < 0.5, parent1, parent2)
#         return offspring

#     # ==================== FAST REPAIR (MINIMAL) ====================
    
#     def quick_repair(self, individual: np.ndarray, func) -> tuple:
#         '''ONE-SHOT repair: remove nodes once, don't iterate.'''
#         solution = individual.copy()
#         fitness = func(solution.tolist())
        
#         if fitness >= 0:
#             return solution, fitness
        
#         # ONE-TIME removal: remove 20% of selected nodes
#         ones_positions = np.where(solution == 1)[0]
        
#         if len(ones_positions) > 0:
#             num_to_remove = max(1, len(ones_positions) // 5)
#             remove_positions = np.random.choice(ones_positions, size=num_to_remove, replace=False)
#             solution[remove_positions] = 0
            
#             # Re-evaluate ONCE
#             if func.state.evaluations < self.budget:
#                 fitness = func(solution.tolist())
        
#         return solution, fitness

#     # ==================== FAST SELECTION ====================
    
#     def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
#         '''Return INDEX only (no copying). Fully vectorized.'''
#         pop_size = len(population)
#         tournament_idx = np.random.choice(
#             pop_size, 
#             size=min(self.tournament_size, pop_size), 
#             replace=False
#         )
        
#         # Vectorized: extract fitness values for tournament
#         tournament_fits = fitnesses[tournament_idx]
#         feasible_mask = tournament_fits >= 0
        
#         if np.any(feasible_mask):
#             # Among feasible: pick best fitness
#             feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
#             best_local = np.argmax(feasible_fits)
#         else:
#             # All infeasible: pick least infeasible
#             best_local = np.argmax(tournament_fits)
        
#         return tournament_idx[best_local]

#     # ==================== MAIN ALGORITHM ====================
    
#     def __call__(self, func: ioh.problem.PBO):
#         n = func.meta_data.n_variables
        
#         # Pre-compute power-law distribution
#         self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         # ==================== ADAPTIVE SPARSE INITIALIZATION ====================
#         population = []
#         for _ in range(self.population_size):
#             individual = np.zeros(n, dtype=int)
            
#             # CRITICAL FIX: Sparse initialization based on problem size
#             if n <= 1000:
#                 # Small problems (MaxCoverage ~450-760): 5-10% density
#                 num_ones = np.random.randint(max(1, n // 20), max(2, n // 10))
#             else:
#                 # Large problems (MaxInfluence ~4000): 1-3% density
#                 num_ones = np.random.randint(max(1, n // 100), max(2, n // 33))
            
#             ones_positions = np.random.choice(n, size=num_ones, replace=False)
#             individual[ones_positions] = 1
#             population.append(individual)
        
#         # Quick initial repair (one-shot, not iterative)
#         fitnesses = np.zeros(self.population_size)
#         for i in range(self.population_size):
#             if func.state.evaluations >= self.budget:
#                 fitnesses = fitnesses[:i]
#                 population = population[:i]
#                 break
            
#             population[i], fitnesses[i] = self.quick_repair(population[i], func)

#         # ==================== SMART EARLY STOPPING ====================
#         best_fitness = np.max(fitnesses) if len(fitnesses) > 0 else -np.inf
#         gens_no_improvement: int = 0
        
#         # Scale patience with budget
#         if self.budget <= 10000:
#                 # Small budget (10K): 15-25% patience
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 150  # 1500 evals patience (15%)
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 100  # 2000 evals patience (20%)
#             else:  # pop_size = 50
#                 max_gens_no_improvement = 50   # 2500 evals patience (25%)
#         else:
#             # 15-20% patience
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 1500  # 15K evals patience (15%)
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 750   # 15K evals patience (15%)
#             else:  # pop_size = 50
#                 max_gens_no_improvement = 400   # 20K evals patience (20%)

#         generation = 0
#         repair_count = 0  # Track repairs per generation
        
#         # ==================== MAIN EVOLUTIONARY LOOP ====================
#         while func.state.evaluations < self.budget:
#             if func.state.optimum_found:
#                 break

#             # Early stopping check
#             if gens_no_improvement >= max_gens_no_improvement:
#                 break
            
#             # Reset repair counter for this generation
#             repair_count = 0

#             # Generate offspring
#             offspring_population = []
#             offspring_fitnesses = []
            
#             for _ in range(self.population_size):
#                 if func.state.evaluations >= self.budget:
#                     break

#                 # OPTIMIZATION: Only do crossover sometimes (controlled by crossover_rate)
#                 if np.random.rand() < self.crossover_rate:
#                     # Crossover path: select 2 parents
#                     parent1_idx = self.tournament_select_fast(population, fitnesses)
#                     parent2_idx = self.tournament_select_fast(population, fitnesses)
#                     parent1 = population[parent1_idx]
#                     parent2 = population[parent2_idx]
#                     offspring = self.uniform_crossover(parent1, parent2)
#                 else:
#                     # Mutation-only path: select 1 parent (FASTER)
#                     parent_idx = self.tournament_select_fast(population, fitnesses)
#                     offspring = population[parent_idx].copy()
                
#                 # Mutate
#                 offspring = self.mutate(offspring, n)
                
#                 # Evaluate
#                 offspring_fitness = func(offspring.tolist())

#                 # ==================== SMART REPAIR STRATEGY ====================
#                 if offspring_fitness < 0:
#                     # Only repair if:
#                     # 1. Haven't repaired too many this generation (max 20%)
#                     # 2. Have enough budget left
#                     # 3. Early in search (< 70% budget used)
#                     should_repair = (
#                         repair_count < self.population_size // 5 and
#                         func.state.evaluations < self.budget - 100 and
#                         func.state.evaluations < self.budget * 0.7
#                     )
                    
#                     if should_repair:
#                         offspring, offspring_fitness = self.quick_repair(offspring, func)
#                         if offspring_fitness >= 0:
#                             repair_count += 1
                
#                 offspring_population.append(offspring)
#                 offspring_fitnesses.append(offspring_fitness)
            
#             if not offspring_population:
#                 break
            
#             # ==================== FAST SURVIVAL SELECTION ====================
#             combined_pop = population + offspring_population
#             offspring_fitnesses = np.array(offspring_fitnesses)
#             combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
#             # Vectorized feasibility check
#             feasible_mask = (combined_fitnesses >= 0).astype(int)
            
#             # Sort by feasibility first, then fitness (descending)
#             sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
#             # Keep top population_size
#             top_indices = sorted_indices[:self.population_size]
#             population = [combined_pop[i] for i in top_indices]
#             fitnesses = combined_fitnesses[top_indices]
            
#             # Track improvement for early stopping
#             current_best = np.max(fitnesses)
#             if current_best > best_fitness:
#                 best_fitness = current_best
#                 # gens_no_improvement = 0
#             else:
#                 gens_no_improvement += 1
            
#             generation += 1


# from .algorithm_interface import Algorithm
# import ioh 
# import numpy as np


# class SingleObjectiveEA(Algorithm):
#     '''
#     Optimized population-based Fast GA using Power Law Mutation for monotone submodular graph problems.
#     Based on the original FastGA with β=1.5 power-law mutation, mutation-only (no crossover).
#     '''
#     def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
#                 tournament_size: int = 3):
#         super().__init__(budget, name="Fast-GA", 
#                         algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta})")
#         self.population_size = population_size
#         self.budget = budget 
#         self.beta = beta
#         self.tournament_size = tournament_size
#         self.power_law_distribution = None

#     # ==================== POWER-LAW MUTATION ====================
    
#     def compute_power_law_distribution(self, n: int) -> np.ndarray:
#         '''Compute power-law distribution (cached).'''
#         distribution = np.zeros(n + 1)
        
#         C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
#         for i in range(1, min(n // 2 + 1, n + 1)):
#             distribution[i] = (i ** (-self.beta)) / C
        
#         distribution[0] = 0.0
#         return distribution

#     def mutate(self, individual: np.ndarray, n: int) -> np.ndarray:
#         '''Fast power-law mutation.'''
#         offspring = individual.copy()
        
#         if self.power_law_distribution is None:
#             self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         k = np.random.choice(n + 1, p=self.power_law_distribution)
#         k = max(1, k)  # At least 1 flip
        
#         flip_indices = np.random.choice(n, size=k, replace=False)
#         offspring[flip_indices] = 1 - offspring[flip_indices]
        
#         return offspring

#     # ==================== FAST REPAIR (MINIMAL) ====================
    
#     def quick_repair(self, individual: np.ndarray, func) -> tuple:
#         '''ONE-SHOT repair: remove nodes once, don't iterate.'''
#         solution = individual.copy()
#         fitness = func(solution.tolist())
        
#         if fitness >= 0:
#             return solution, fitness
        
#         # ONE-TIME removal: remove 20% of selected nodes
#         ones_positions = np.where(solution == 1)[0]
        
#         if len(ones_positions) > 0:
#             num_to_remove = max(1, len(ones_positions) // 5)
#             remove_positions = np.random.choice(ones_positions, size=num_to_remove, replace=False)
#             solution[remove_positions] = 0
            
#             # Re-evaluate ONCE
#             if func.state.evaluations < self.budget:
#                 fitness = func(solution.tolist())
        
#         return solution, fitness

#     # ==================== FAST SELECTION ====================
    
#     def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
#         '''Return INDEX only (no copying). Fully vectorized.'''
#         pop_size = len(population)
#         tournament_idx = np.random.choice(
#             pop_size, 
#             size=min(self.tournament_size, pop_size), 
#             replace=False
#         )
        
#         # Vectorized: extract fitness values for tournament
#         tournament_fits = fitnesses[tournament_idx]
#         feasible_mask = tournament_fits >= 0
        
#         if np.any(feasible_mask):
#             # Among feasible: pick best fitness
#             feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
#             best_local = np.argmax(feasible_fits)
#         else:
#             # All infeasible: pick least infeasible
#             best_local = np.argmax(tournament_fits)
        
#         return tournament_idx[best_local]

#     # ==================== MAIN ALGORITHM ====================
    
#     def __call__(self, func: ioh.problem.PBO):
#         n = func.meta_data.n_variables
        
#         # Pre-compute power-law distribution
#         self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         # ==================== ULTRA-SPARSE INITIALIZATION ====================
#         population = []
#         for _ in range(self.population_size):
#             individual = np.zeros(n, dtype=int)
            
#             # CRITICAL: Much sparser initialization
#             if n <= 1000:
#                 # Small problems (MaxCoverage ~450-760): 3-5% density
#                 num_ones = np.random.randint(max(1, n // 33), max(2, n // 20))
#             else:
#                 # Large problems (MaxInfluence ~4000): 0.5-1% density (MUCH SPARSER!)
#                 num_ones = np.random.randint(max(1, n // 200), max(2, n // 100))
#                 # For n=4039: 20-40 nodes (was 40-122!)
            
#             ones_positions = np.random.choice(n, size=num_ones, replace=False)
#             individual[ones_positions] = 1
#             population.append(individual)
        
#         # Quick initial repair (one-shot, not iterative)
#         fitnesses = np.zeros(self.population_size)
#         for i in range(self.population_size):
#             if func.state.evaluations >= self.budget:
#                 fitnesses = fitnesses[:i]
#                 population = population[:i]
#                 break
            
#             population[i], fitnesses[i] = self.quick_repair(population[i], func)

#         # ==================== AGGRESSIVE EARLY STOPPING ====================
#         best_fitness = np.max(fitnesses) if len(fitnesses) > 0 else -np.inf
#         gens_no_improvement: int = 0
        
#         # MUCH MORE AGGRESSIVE patience (10-15% of budget)
#         if self.budget <= 10000:
#             # Exercise 3: VERY aggressive early stopping
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 100  # 1000 evals patience (10%)
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 60   # 1200 evals patience (12%)
#             else:  # pop_size = 50
#                 max_gens_no_improvement = 30   # 1500 evals patience (15%)
#         else:
#             # Exercise 4: Still aggressive but more patient
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 1000  # 10K evals patience (10%)
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 500   # 10K evals patience (10%)
#             else:  # pop_size = 50
#                 max_gens_no_improvement = 250   # 12.5K evals patience (12.5%)

#         generation = 0
#         repair_count = 0
        
#         # Track when we last improved (for logging)
#         last_improvement_gen = 0
        
#         # ==================== MAIN EVOLUTIONARY LOOP ====================
#         while func.state.evaluations < self.budget:
#             if func.state.optimum_found:
#                 break

#             # Early stopping check
#             if gens_no_improvement >= max_gens_no_improvement:
#                 print(f"[EARLY STOP] Gen {generation}: No improvement for {gens_no_improvement} gens "
#                     f"(last improvement at gen {last_improvement_gen})")
#                 break
            
#             # Reset repair counter for this generation
#             repair_count = 0

#             # Generate offspring
#             offspring_population = []
#             offspring_fitnesses = []
            
#             for _ in range(self.population_size):
#                 if func.state.evaluations >= self.budget:
#                     break

#                 # MUTATION-ONLY: Select 1 parent
#                 parent_idx = self.tournament_select_fast(population, fitnesses)
#                 parent = population[parent_idx]
                
#                 # Mutate
#                 offspring = self.mutate(parent, n)
                
#                 # Evaluate
#                 offspring_fitness = func(offspring.tolist())

#                 # ==================== VERY LIMITED REPAIR ====================
#                 if offspring_fitness < 0:
#                     # Only repair if:
#                     # 1. Very few repairs this gen (max 1-2)
#                     # 2. Have budget left
#                     # 3. VERY early in search (< 50% budget)
#                     should_repair = (
#                         repair_count < max(1, self.population_size // 10) and  # Max 1-2 repairs
#                         func.state.evaluations < self.budget - 50 and
#                         func.state.evaluations < self.budget * 0.5  # Only first 50%
#                     )
                    
#                     if should_repair:
#                         offspring, offspring_fitness = self.quick_repair(offspring, func)
#                         if offspring_fitness >= 0:
#                             repair_count += 1
                
#                 offspring_population.append(offspring)
#                 offspring_fitnesses.append(offspring_fitness)
            
#             if not offspring_population:
#                 break
            
#             # ==================== FAST SURVIVAL SELECTION ====================
#             combined_pop = population + offspring_population
#             offspring_fitnesses = np.array(offspring_fitnesses)
#             combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
#             # Vectorized feasibility check
#             feasible_mask = (combined_fitnesses >= 0).astype(int)
            
#             # Sort by feasibility first, then fitness (descending)
#             sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
#             # Keep top population_size
#             top_indices = sorted_indices[:self.population_size]
#             population = [combined_pop[i] for i in top_indices]
#             fitnesses = combined_fitnesses[top_indices]
            
#             # ==================== TRACK IMPROVEMENT (ABSOLUTE) ====================
#             current_best = np.max(fitnesses)
#             if current_best > best_fitness:
#                 best_fitness = current_best
#                 last_improvement_gen = generation
#                 # ABSOLUTE stopping: DON'T reset counter
#             else:
#                 gens_no_improvement += 1
            
#             generation += 1



from .algorithm_interface import Algorithm
import ioh 
import numpy as np


# class SingleObjectiveEA(Algorithm):
#     '''
#     Optimized population-based Fast GA using Power Law Mutation for monotone submodular graph problems.
#     Based on the original FastGA with β=1.5 power-law mutation, mutation-only (no crossover).
#     '''
#     def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
#                 tournament_size: int = 3):
#         super().__init__(budget, name="Fast-GA", 
#                         algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta})")
#         self.population_size = population_size
#         self.budget = budget 
#         self.beta = beta
#         self.tournament_size = tournament_size
#         self.power_law_distribution = None

#     # ==================== POWER-LAW MUTATION ====================
    
#     def compute_power_law_distribution(self, n: int) -> np.ndarray:
#         '''Compute power-law distribution (cached).'''
#         distribution = np.zeros(n + 1)
        
#         C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
#         for i in range(1, min(n // 2 + 1, n + 1)):
#             distribution[i] = (i ** (-self.beta)) / C
        
#         distribution[0] = 0.0
#         return distribution

#     def mutate(self, individual: np.ndarray, n: int, generation: int, max_patience: int) -> np.ndarray:
#         '''
#         Adaptive power-law mutation with diversity boost.
#         '''
#         offspring = individual.copy()
        
#         if self.power_law_distribution is None:
#             self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         # Sample from power-law distribution
#         k = np.random.choice(n + 1, p=self.power_law_distribution)
#         k = max(1, k)  # At least 1 flip
        
#         # DIVERSITY BOOST: After 60% progress, 20% chance of stronger mutation
#         progress = generation / max(1, max_patience)
#         if progress > 0.6 and np.random.random() < 0.2:
#             # Stronger mutation: 2-4x the sampled k
#             k = min(n, k * np.random.randint(2, 5))
        
#         flip_indices = np.random.choice(n, size=k, replace=False)
#         offspring[flip_indices] = 1 - offspring[flip_indices]
        
#         return offspring

#     # ==================== FAST SELECTION ====================
    
#     def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
#         '''Return INDEX only (no copying). Fully vectorized.'''
#         pop_size = len(population)
#         tournament_idx = np.random.choice(
#             pop_size, 
#             size=min(self.tournament_size, pop_size), 
#             replace=False
#         )
        
#         # Vectorized: extract fitness values for tournament
#         tournament_fits = fitnesses[tournament_idx]
#         feasible_mask = tournament_fits >= 0
        
#         if np.any(feasible_mask):
#             # Among feasible: pick best fitness
#             feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
#             best_local = np.argmax(feasible_fits)
#         else:
#             # All infeasible: pick least infeasible
#             best_local = np.argmax(tournament_fits)
        
#         return tournament_idx[best_local]

#     # ==================== MAIN ALGORITHM ====================
    
#     def __call__(self, func: ioh.problem.PBO):
#         n = func.meta_data.n_variables
        
#         # Pre-compute power-law distribution
#         self.power_law_distribution = self.compute_power_law_distribution(n)
        
#         # ==================== ULTRA-SAFE INITIALIZATION ====================
#         # CRITICAL: Start with VERY FEW nodes to guarantee feasibility
#         # For B=10, start with just 4-6 nodes (well under budget)
        
#         population = []
#         fitnesses = []
        
#         for _ in range(self.population_size):
#             individual = np.zeros(n, dtype=int)
            
#             # SAFE INITIALIZATION: 4-6 nodes (well under B=10)
#             num_ones = np.random.randint(4, 7)
            
#             ones_positions = np.random.choice(n, size=num_ones, replace=False)
#             individual[ones_positions] = 1
            
#             # Direct evaluation (NO REPAIR during initialization)
#             fitness = func(individual.tolist())
            
#             population.append(individual)
#             fitnesses.append(fitness)
            
#             if func.state.evaluations >= self.budget:
#                 break
        
#         fitnesses = np.array(fitnesses)

#         # ==================== RELAXED EARLY STOPPING ====================
#         # YOUR SUGGESTION: Don't start counting until 65% budget used!
        
#         feasible_fitnesses = fitnesses[fitnesses >= 0]
#         best_feasible_fitness = np.max(feasible_fitnesses) if len(feasible_fitnesses) > 0 else -np.inf
#         gens_no_improvement: int = 0
        
#         # Start counting NO-IMPROVEMENT only after 65% budget used
#         budget_threshold_for_stopping = int(self.budget * 0.65)
        
#         # VERY RELAXED patience: Allow more exploration
#         if self.budget <= 10000:
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 200  # 2000 evals patience after 65% mark
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 150  # 3000 evals patience after 65% mark
#             else:  # pop_size = 50
#                 max_gens_no_improvement = 80   # 4000 evals patience after 65% mark
#         else:
#             if self.population_size <= 10:
#                 max_gens_no_improvement = 2000
#             elif self.population_size <= 20:
#                 max_gens_no_improvement = 1500
#             else:
#                 max_gens_no_improvement = 1000

#         generation = 0
#         last_improvement_gen = 0
#         last_improvement_evals = 0
        
#         # ==================== MAIN EVOLUTIONARY LOOP ====================
#         while func.state.evaluations < self.budget:
#             if func.state.optimum_found:
#                 print(f"[OPTIMUM FOUND] Gen {generation}, Evals {func.state.evaluations}")
#                 break

#             # Early stopping check (ONLY after 65% budget used)
#             if func.state.evaluations >= budget_threshold_for_stopping:
#                 if gens_no_improvement >= max_gens_no_improvement:
#                     print(f"[EARLY STOP] Gen {generation}, Evals {func.state.evaluations}: "
#                         f"No improvement for {gens_no_improvement} gens after 65% budget "
#                         f"(last improvement: gen {last_improvement_gen}, eval {last_improvement_evals})")
#                     break

#             # Generate offspring
#             offspring_population = []
#             offspring_fitnesses = []
            
#             for _ in range(self.population_size):
#                 if func.state.evaluations >= self.budget:
#                     break

#                 # MUTATION-ONLY: Select 1 parent
#                 parent_idx = self.tournament_select_fast(population, fitnesses)
#                 parent = population[parent_idx]
                
#                 # Mutate with adaptive strength
#                 offspring = self.mutate(parent, n, generation, max_gens_no_improvement)
                
#                 # Evaluate (NO REPAIR - let evolution handle it)
#                 offspring_fitness = func(offspring.tolist())
                
#                 offspring_population.append(offspring)
#                 offspring_fitnesses.append(offspring_fitness)
            
#             if not offspring_population:
#                 break
            
#             # ==================== FAST SURVIVAL SELECTION ====================
#             combined_pop = population + offspring_population
#             offspring_fitnesses = np.array(offspring_fitnesses)
#             combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
#             # Vectorized feasibility check
#             feasible_mask = (combined_fitnesses >= 0).astype(int)
            
#             # Sort by feasibility first, then fitness (descending)
#             sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
#             # Keep top population_size
#             top_indices = sorted_indices[:self.population_size]
#             population = [combined_pop[i] for i in top_indices]
#             fitnesses = combined_fitnesses[top_indices]
            
#             # ==================== TRACK FEASIBLE IMPROVEMENT ====================
#             # Only start counting after 65% budget used
#             current_feasible = fitnesses[fitnesses >= 0]
#             if len(current_feasible) > 0:
#                 current_best_feasible = np.max(current_feasible)
                
#                 # Check for improvement
#                 if current_best_feasible > best_feasible_fitness:
#                     best_feasible_fitness = current_best_feasible
#                     last_improvement_gen = generation
#                     last_improvement_evals = func.state.evaluations
#                     gens_no_improvement = 0  # RESET
#                 else:
#                     # Only increment if past 65% budget threshold
#                     if func.state.evaluations >= budget_threshold_for_stopping:
#                         gens_no_improvement += 1
#             else:
#                 # No feasible solutions yet (shouldn't happen with safe init)
#                 if func.state.evaluations >= budget_threshold_for_stopping:
#                     gens_no_improvement += 1
            
#             generation += 1

#         # ==================== ENSURE FINAL SOLUTION IS FEASIBLE ====================
#         feasible_indices = np.where(fitnesses >= 0)[0]
        
#         if len(feasible_indices) > 0:
#             best_feasible_idx = feasible_indices[np.argmax(fitnesses[feasible_indices])]
#             final_best = population[best_feasible_idx]
#             final_fitness = fitnesses[best_feasible_idx]
#         else:
#             # Emergency: pick best overall
#             best_idx = np.argmax(fitnesses)
#             final_best = population[best_idx]
#             final_fitness = fitnesses[best_idx]
        
#         # Final evaluation to register with IOH
#         if func.state.evaluations < self.budget:
#             final_fitness = func(final_best.tolist())
        
#         print(f"[FINAL] Gen {generation}, Evals {func.state.evaluations}: "
#              f"Best feasible fitness = {final_fitness:.2f}")
        
#         return final_best




class SingleObjectiveEA(Algorithm):
    '''
    Population-based Fast GA with diversity-preserving restarts for submodular optimization.
    '''
    def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
                tournament_size: int = 3):
        super().__init__(budget, name="Fast-GA", 
                        algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta})")
        self.population_size = population_size
        self.budget = budget 
        self.beta = beta
        self.tournament_size = min(tournament_size + 2, population_size // 2)
        self.power_law_distribution = None

    def compute_power_law_distribution(self, n: int) -> np.ndarray:
        '''Compute power-law distribution (cached).'''
        distribution = np.zeros(n + 1)
        C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        for i in range(1, min(n // 2 + 1, n + 1)):
            distribution[i] = (i ** (-self.beta)) / C
        distribution[0] = 0.0
        return distribution

    def mutate(self, individual: np.ndarray, n: int, strength: str = 'normal') -> np.ndarray:
        '''Power-law mutation with controllable strength.'''
        offspring = individual.copy()
        
        if self.power_law_distribution is None:
            self.power_law_distribution = self.compute_power_law_distribution(n)
        
        k = np.random.choice(n + 1, p=self.power_law_distribution)
        k = max(1, k)
        
        if strength == 'strong':
            k = min(n, k * np.random.randint(2, 4))
        elif strength == 'very_strong':
            k = min(n, k * np.random.randint(3, 6))
        
        flip_indices = np.random.choice(n, size=k, replace=False)
        offspring[flip_indices] = 1 - offspring[flip_indices]
        
        return offspring

    def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
        '''Return INDEX only.'''
        pop_size = len(population)
        tournament_idx = np.random.choice(
            pop_size, 
            size=min(self.tournament_size, pop_size), 
            replace=False
        )
        
        tournament_fits = fitnesses[tournament_idx]
        feasible_mask = tournament_fits >= 0
        
        if np.any(feasible_mask):
            feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
            best_local = np.argmax(feasible_fits)
        else:
            best_local = np.argmax(tournament_fits)
        
        return tournament_idx[best_local]

    def check_diversity(self, population: list) -> float:
        '''Calculate population diversity (average Hamming distance).'''
        if len(population) < 2:
            return 1.0
        
        num_samples = min(50, len(population) * (len(population) - 1) // 2)
        distances = []
        
        for _ in range(num_samples):
            i, j = np.random.choice(len(population), size=2, replace=False)
            distance = np.sum(population[i] != population[j]) / len(population[i])
            distances.append(distance)
        
        return np.mean(distances)

    def restart_population(self, population: list, fitnesses: np.ndarray, n: int, func) -> tuple:
        '''Diversity restart: Keep best 20% + generate fresh 80%.'''
        num_elite = max(1, self.population_size // 5)
        
        feasible_mask = fitnesses >= 0
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            elite_indices = feasible_indices[np.argsort(fitnesses[feasible_indices])[-num_elite:]]
        else:
            elite_indices = np.argsort(fitnesses)[-num_elite:]
        
        new_population = [population[i].copy() for i in elite_indices]
        new_fitnesses = [fitnesses[i] for i in elite_indices]
        
        for _ in range(self.population_size - num_elite):
            if func.state.evaluations >= self.budget:
                break
            
            individual = np.zeros(n, dtype=int)
            num_ones = np.random.randint(4, 9)
            ones_positions = np.random.choice(n, size=num_ones, replace=False)
            individual[ones_positions] = 1
            
            fitness = func(individual.tolist())
            new_population.append(individual)
            new_fitnesses.append(fitness)
        
        return new_population, np.array(new_fitnesses)

    def __call__(self, func: ioh.problem.PBO):
        n = func.meta_data.n_variables
        self.power_law_distribution = self.compute_power_law_distribution(n)
        
        # ==================== INITIAL POPULATION ====================
        population = []
        fitnesses = []
        
        for _ in range(self.population_size):
            individual = np.zeros(n, dtype=int)
            num_ones = np.random.randint(4, 7)
            ones_positions = np.random.choice(n, size=num_ones, replace=False)
            individual[ones_positions] = 1
            
            fitness = func(individual.tolist())
            population.append(individual)
            fitnesses.append(fitness)
            
            if func.state.evaluations >= self.budget:
                break
        
        fitnesses = np.array(fitnesses)

        # ==================== BUDGET THRESHOLDS ====================
        # CRITICAL: 70% budget threshold (7000 evals for 10K budget)
        budget_threshold_for_counting = int(self.budget * 0.70)
        
        print(f"[INIT] Budget={self.budget}, Will start counting no-improvement after {budget_threshold_for_counting} evals (70%)")
        
        feasible_fitnesses = fitnesses[fitnesses >= 0]
        best_feasible_fitness = np.max(feasible_fitnesses) if len(feasible_fitnesses) > 0 else -np.inf
        
        # Counter that only increments AFTER budget threshold
        gens_no_improvement = 0
        
        if self.budget <= 10000:
            if self.population_size <= 10:
                max_gens_no_improvement = 200
                restart_threshold = 80
            elif self.population_size <= 20:
                max_gens_no_improvement = 150
                restart_threshold = 60
            else:
                max_gens_no_improvement = 100
                restart_threshold = 40
        else:
            if self.population_size <= 10:
                max_gens_no_improvement = 2000
                restart_threshold = 800
            elif self.population_size <= 20:
                max_gens_no_improvement = 1500
                restart_threshold = 600
            else:
                max_gens_no_improvement = 1000
                restart_threshold = 400

        generation = 0
        last_improvement_gen = 0
        last_improvement_evals = 0
        num_restarts = 0
        
        # ==================== MAIN LOOP ====================
        while func.state.evaluations < self.budget:
            if func.state.optimum_found:
                print(f"[OPTIMUM FOUND] Gen {generation}, Evals {func.state.evaluations}")
                break

            # ==================== EARLY CHECKS (ONLY AFTER 70% BUDGET) ====================
            # CRITICAL: All stopping/restart logic ONLY after threshold
            if func.state.evaluations >= budget_threshold_for_counting:
                
                # Restart check
                if gens_no_improvement >= restart_threshold and num_restarts < 2:
                    diversity = self.check_diversity(population)
                    
                    if diversity < 0.15 and func.state.evaluations < self.budget * 0.90:
                        print(f"[RESTART] Gen {generation}, Evals {func.state.evaluations}: "
                              f"Diversity {diversity:.3f}, gens_no_improvement={gens_no_improvement}, "
                              f"restarting (#{num_restarts + 1})")
                        
                        population, fitnesses = self.restart_population(population, fitnesses, n, func)
                        gens_no_improvement = 0
                        num_restarts += 1
                        
                        feasible_fitnesses = fitnesses[fitnesses >= 0]
                        if len(feasible_fitnesses) > 0:
                            best_feasible_fitness = max(best_feasible_fitness, np.max(feasible_fitnesses))

                # Early stop check
                if gens_no_improvement >= max_gens_no_improvement:
                    print(f"[EARLY STOP] Gen {generation}, Evals {func.state.evaluations}: "
                          f"No improvement for {gens_no_improvement} gens (started counting at {budget_threshold_for_counting} evals) "
                          f"(last improvement: gen {last_improvement_gen}, eval {last_improvement_evals})")
                    break

            # ==================== OFFSPRING GENERATION ====================
            offspring_population = []
            offspring_fitnesses = []
            
            progress = func.state.evaluations / self.budget
            
            for _ in range(self.population_size):
                if func.state.evaluations >= self.budget:
                    break

                parent_idx = self.tournament_select_fast(population, fitnesses)
                parent = population[parent_idx]
                
                if progress < 0.4:
                    mutation_strength = 'normal'
                elif progress < 0.7:
                    mutation_strength = 'strong' if np.random.random() < 0.2 else 'normal'
                else:
                    if np.random.random() < 0.3:
                        mutation_strength = 'very_strong'
                    elif np.random.random() < 0.5:
                        mutation_strength = 'strong'
                    else:
                        mutation_strength = 'normal'
                
                offspring = self.mutate(parent, n, strength=mutation_strength)
                offspring_fitness = func(offspring.tolist())
                
                offspring_population.append(offspring)
                offspring_fitnesses.append(offspring_fitness)
            
            if not offspring_population:
                break
            
            # ==================== SURVIVAL SELECTION ====================
            combined_pop = population + offspring_population
            offspring_fitnesses = np.array(offspring_fitnesses)
            combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
            feasible_mask = (combined_fitnesses >= 0).astype(int)
            sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
            top_indices = sorted_indices[:self.population_size]
            population = [combined_pop[i] for i in top_indices]
            fitnesses = combined_fitnesses[top_indices]
            
            # ==================== TRACK IMPROVEMENT ====================
            current_feasible = fitnesses[fitnesses >= 0]
            if len(current_feasible) > 0:
                current_best_feasible = np.max(current_feasible)
                
                if current_best_feasible > best_feasible_fitness:
                    best_feasible_fitness = current_best_feasible
                    last_improvement_gen = generation
                    last_improvement_evals = func.state.evaluations
                    gens_no_improvement = 0  # ALWAYS reset (even before threshold)
                else:
                    # CRITICAL: ONLY increment if past 70% budget threshold
                    if func.state.evaluations >= budget_threshold_for_counting:
                        gens_no_improvement += 1
            else:
                # No feasible solutions
                if func.state.evaluations >= budget_threshold_for_counting:
                    gens_no_improvement += 1
            
            generation += 1

        # ==================== FINAL SOLUTION ====================
        feasible_indices = np.where(fitnesses >= 0)[0]
        
        if len(feasible_indices) > 0:
            best_feasible_idx = feasible_indices[np.argmax(fitnesses[feasible_indices])]
            final_best = population[best_feasible_idx]
            final_fitness = fitnesses[best_feasible_idx]
        else:
            best_idx = np.argmax(fitnesses)
            final_best = population[best_idx]
            final_fitness = fitnesses[best_idx]
        
        if func.state.evaluations < self.budget:
            final_fitness = func(final_best.tolist())
        
        print(f"[FINAL] Gen {generation}, Evals {func.state.evaluations}: "
              f"Best feasible fitness = {final_fitness:.2f} (restarts: {num_restarts})")
        
        return final_best