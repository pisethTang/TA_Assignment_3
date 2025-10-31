from .algorithm_interface import Algorithm
import ioh 
import numpy as np



class SingleObjectiveEA(Algorithm):
    '''
    Optimized population-based Fast GA for monotone submodular graph problems.
    Fast version with minimal repairs and early stopping.
    '''
    def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
                tournament_size: int = 3):
        super().__init__(budget, name="Fast-GA", 
                        algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta})")
        self.population_size = population_size
        self.budget = budget 
        self.beta = beta
        self.tournament_size = tournament_size
        self.power_law_distribution = None

    # ==================== POWER-LAW MUTATION ====================
    
    def compute_power_law_distribution(self, n: int) -> np.ndarray:
        '''Compute power-law distribution (cached).'''
        distribution = np.zeros(n + 1)
        
        C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
        for i in range(1, min(n // 2 + 1, n + 1)):
            distribution[i] = (i ** (-self.beta)) / C
        
        distribution[0] = 0.0
        return distribution

    def mutate(self, individual: np.ndarray, n: int) -> np.ndarray:
        '''Fast power-law mutation.'''
        offspring = individual.copy()
        
        if self.power_law_distribution is None:
            self.power_law_distribution = self.compute_power_law_distribution(n)
        
        k = np.random.choice(n + 1, p=self.power_law_distribution)
        k = max(1, k)  # At least 1 flip
        
        flip_indices = np.random.choice(n, size=k, replace=False)
        offspring[flip_indices] = 1 - offspring[flip_indices]
        
        return offspring

    # ==================== FAST REPAIR (MINIMAL) ====================
    
    def quick_repair(self, individual: np.ndarray, func) -> tuple:
        '''
        ONE-SHOT repair: remove nodes once, don't iterate.
        Much faster than your version.
        '''
        solution = individual.copy()
        fitness = func(solution.tolist())
        
        if fitness >= 0:
            return solution, fitness
        
        # ONE-TIME removal: remove 20% of selected nodes
        ones_positions = np.where(solution == 1)[0]
        
        if len(ones_positions) > 0:
            num_to_remove = max(1, len(ones_positions) // 5)
            remove_positions = np.random.choice(ones_positions, size=num_to_remove, replace=False)
            solution[remove_positions] = 0
            
            # Re-evaluate ONCE
            if func.state.evaluations < self.budget:
                fitness = func(solution.tolist())
        
        return solution, fitness

    # ==================== FAST SELECTION ====================
    
    def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
        '''
        Return INDEX only (no copying). Fully vectorized.
        '''
        pop_size = len(population)
        tournament_idx = np.random.choice(
            pop_size, 
            size=min(self.tournament_size, pop_size), 
            replace=False
        )
        
        # Vectorized: extract fitness values for tournament
        tournament_fits = fitnesses[tournament_idx]
        feasible_mask = tournament_fits >= 0
        
        if np.any(feasible_mask):
            # Among feasible: pick best fitness
            feasible_fits = np.where(feasible_mask, tournament_fits, -np.inf)
            best_local = np.argmax(feasible_fits)
        else:
            # All infeasible: pick least infeasible
            best_local = np.argmax(tournament_fits)
        
        return tournament_idx[best_local]

    # ==================== MAIN ALGORITHM ====================
    
    def __call__(self, func: ioh.problem.PBO):
        n = func.meta_data.n_variables
        
        # Pre-compute power-law distribution
        self.power_law_distribution = self.compute_power_law_distribution(n)
        
        # Sparse initialization
        population = []
        for _ in range(self.population_size):
            individual = np.zeros(n, dtype=int)
            # with 2-5% of nodes (more likely to be feasible)
            num_ones = np.random.randint(max(1, n // 50), max(2, n // 20))
            ones_positions = np.random.choice(n, size=num_ones, replace=False)
            individual[ones_positions] = 1
            population.append(individual)
        
        # Quick initial repair (one-shot, not iterative)
        fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            if func.state.evaluations >= self.budget:
                fitnesses = fitnesses[:i]  # Truncate if budget ran out
                population = population[:i]
                break
            
            population[i], fitnesses[i] = self.quick_repair(population[i], func)

        # Early stopping
        best_fitness = np.max(fitnesses) if len(fitnesses) > 0 else -np.inf
        gens_no_improvement = 0
        max_gens_no_improvement = 30  # Stop if no improvement for 30 gens
        
        generation = 0
        
        # Main evolutionary loop
        while func.state.evaluations < self.budget:
            if func.state.optimum_found:
                break

            # Early stopping check
            if gens_no_improvement >= max_gens_no_improvement:
                break
            
            # Track feasibility for adaptive repair (vectorized)
            num_feasible = np.sum(fitnesses >= 0)
            
            # Generate offspring
            offspring_population = []
            offspring_fitnesses = []
            
            for _ in range(self.population_size):
                if func.state.evaluations >= self.budget:
                    break

                # Index-based selection (no copying)
                parent_idx = self.tournament_select_fast(population, fitnesses)
                parent = population[parent_idx]
                
                # Mutate
                offspring = self.mutate(parent, n)
                
                # Evaluate
                offspring_fitness = func(offspring.tolist())

                # Minimal repair (only if needed)
                if offspring_fitness < 0:
                    # Only repair if:
                    # 1. Less than half the population is feasible, OR
                    # 2. We have enough budget left
                    should_repair = (
                        num_feasible < self.population_size // 2 and 
                        func.state.evaluations < self.budget - 20
                    )
                    
                    if should_repair:
                        offspring, offspring_fitness = self.quick_repair(offspring, func)
                
                offspring_population.append(offspring)
                offspring_fitnesses.append(offspring_fitness)
            
            if not offspring_population:
                break
            
            # Fast survival selection with NumPy
            combined_pop = population + offspring_population


            offspring_fitnesses = np.array(offspring_fitnesses)
            combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
            # Vectorized feasibility check
            feasible_mask = (combined_fitnesses >= 0).astype(int)
            
            # lexsort sorts by keys from LAST to FIRST (bottom to top)
            # So we pass: (fitness, feasibility) to sort by feasibility first, then fitness
            # Negate values to get descending order
            sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
            # Keep top population_size
            top_indices = sorted_indices[:self.population_size]
            population = [combined_pop[i] for i in top_indices]
            fitnesses = combined_fitnesses[top_indices]  # Keep as NumPy array
            
            # Track improvement for early stopping (vectorized)
            current_best = np.max(fitnesses)
            if current_best > best_fitness: # Improvement
                best_fitness = current_best
                gens_no_improvement = 0    
            else:                           # No improvement
                gens_no_improvement += 1
            
            generation += 1