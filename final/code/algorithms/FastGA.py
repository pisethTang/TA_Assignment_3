from .algorithm_interface import Algorithm
import ioh 
import numpy as np


from .algorithm_interface import Algorithm
import ioh 
import numpy as np


class FastGA(Algorithm):
    '''
    Optimized population-based Fast GA with proper all-time best tracking.
    '''
    def __init__(self, budget: int, population_size: int = 20, beta: float = 1.5, 
                tournament_size: int = 3, patience_factor: float = 0.04):
        super().__init__(budget, name=f"Fast-GA-{int(budget/1000)}K-{population_size}", 
                        algorithm_info=f"Population-based Fast-GA (pop={population_size}, β={beta})")
        self.population_size = population_size
        self.budget = budget 
        self.beta = beta
        self.tournament_size = tournament_size
        self.power_law_distribution = None
        self.patience_evaluations = int(budget * patience_factor)
        
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
        k = max(1, k)
        
        flip_indices = np.random.choice(n, size=k, replace=False)
        offspring[flip_indices] = 1 - offspring[flip_indices]
        
        return offspring

    def quick_repair(self, individual: np.ndarray, func) -> tuple:
        '''ONE-SHOT repair: remove nodes once.'''
        solution = individual.copy()
        fitness = func(solution.tolist())
        
        if fitness >= 0:
            return solution, fitness
        
        ones_positions = np.where(solution == 1)[0]
        
        if len(ones_positions) > 0:
            num_to_remove = max(1, len(ones_positions) // 5)
            remove_positions = np.random.choice(ones_positions, size=num_to_remove, replace=False)
            solution[remove_positions] = 0
            
            if func.state.evaluations < self.budget:
                fitness = func(solution.tolist())
        
        return solution, fitness

    def tournament_select_fast(self, population: list, fitnesses: np.ndarray) -> int:
        '''Return INDEX only. Fully vectorized.'''
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

    def initialize_population(self, n: int, func) -> tuple:
        '''Ultra-safe initialization with very few nodes.'''
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
        return population, fitnesses

    def __call__(self, func: ioh.problem.PBO):
        n = func.meta_data.n_variables
        
        # Pre-compute power-law distribution
        self.power_law_distribution = self.compute_power_law_distribution(n)
        
        # Initialize population
        population, fitnesses = self.initialize_population(n, func)
   
        # Track ALL-TIME best fitness (not just current population best)
        all_time_best_fitness = np.max(fitnesses) if len(fitnesses) > 0 else -np.inf
        all_time_best_individual = population[np.argmax(fitnesses)].copy() if len(fitnesses) > 0 else None
        evals_at_last_improvement = func.state.evaluations
        generation = 0
        
        # Main evolutionary loop
        while func.state.evaluations < self.budget:
            if func.state.optimum_found:
                # print(f"[EARLY STOP] Optimum found at Gen {generation}, Evals {func.state.evaluations}")
                break

            # EARLY STOPPING based on ALL-TIME best (not current population)
            evals_since_improvement = func.state.evaluations - evals_at_last_improvement
            if evals_since_improvement >= self.patience_evaluations:
                # print(f"[EARLY STOP] No improvement for {evals_since_improvement} evals. "
                    #   f"Stopping at Gen {generation}, Evals {func.state.evaluations}, "
                    #   f"All-time best fitness: {all_time_best_fitness:.2f}")
                break
            
            # Generate offspring
            offspring_population = []
            offspring_fitnesses = []
            
            for _ in range(self.population_size):
                if func.state.evaluations >= self.budget:
                    break

                parent_idx = self.tournament_select_fast(population, fitnesses)
                parent = population[parent_idx]
                offspring = self.mutate(parent, n)
                offspring_fitness = func(offspring.tolist())

                # Minimal repair
                if offspring_fitness < 0 and func.state.evaluations < self.budget - 10:
                    offspring, offspring_fitness = self.quick_repair(offspring, func)

                offspring_population.append(offspring)
                offspring_fitnesses.append(offspring_fitness)
                
                # Track ALL-TIME best (even before survival selection)
                if offspring_fitness > all_time_best_fitness:
                    all_time_best_fitness = offspring_fitness
                    all_time_best_individual = offspring.copy()
                    evals_at_last_improvement = func.state.evaluations
            
            if not offspring_population:
                break
            
            # Fast survival selection
            combined_pop = population + offspring_population
            offspring_fitnesses = np.array(offspring_fitnesses)
            combined_fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
            feasible_mask = (combined_fitnesses >= 0).astype(int)
            sorted_indices = np.lexsort((-combined_fitnesses, -feasible_mask))
            
            top_indices = sorted_indices[:self.population_size]
            population = [combined_pop[i] for i in top_indices]
            fitnesses = combined_fitnesses[top_indices]
            
            # Also check if current population has better fitness than all-time best
            current_best = np.max(fitnesses)
            if current_best > all_time_best_fitness:
                all_time_best_fitness = current_best
                all_time_best_individual = population[np.argmax(fitnesses)].copy()
                evals_at_last_improvement = func.state.evaluations
            
            generation += 1
        
        # Report final statistics with ALL-TIME best
        # print(f"[FINAL] Gen {generation}, Evals {func.state.evaluations}, "
            #   f"All-time best fitness: {all_time_best_fitness:.2f}")