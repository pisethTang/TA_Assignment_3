import ioh 
from .algorithm_interface import Algorithm
import numpy as np
import random


class MultiObjectiveEA(Algorithm):
    """
    SPEA2-inspired Multi-Objective EA for submodular optimization.
    
    Design choices:
    1. Dual population: main population + external archive
    2. Objectives: Maximize fitness, Minimize cardinality
    3. Fitness assignment: Strength + density estimation
    4. Archive truncation: Remove crowded solutions
    5. Power-law mutation for submodular problems
    """
    
    def __init__(self, budget: int,
                 name: str = "SPEA2-SubMod",
                 algorithm_info: str = "SPEA2-inspired algorithm for submodular problems",
                 population_size: int = 10,
                 archive_size: int = 10,
                 beta: float = 1.5):
        super().__init__(budget, name=name, algorithm_info=algorithm_info)
        self.population_size = population_size
        self.archive_size = archive_size
        self.beta = beta  # Power-law mutation parameter
        self.power_law_distribution = None
    
    # ==================== DOMINANCE ====================
    
    def dominates(self, x_fit, x_card, y_fit, y_card):
        """
        Check if x dominates y.
        Objectives: Maximize fitness, Minimize cardinality
        """
        better_fit = x_fit >= y_fit
        better_card = x_card <= y_card
        strictly_better = (x_fit > y_fit) or (x_card < y_card)
        
        return better_fit and better_card and strictly_better
    
    # ==================== FITNESS ASSIGNMENT ====================
    
    def compute_strength(self, population, fitnesses, cardinalities):
        """Compute strength values: how many solutions does each dominate?"""
        n = len(population)
        strengths = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and self.dominates(fitnesses[i], cardinalities[i], 
                                            fitnesses[j], cardinalities[j]):
                    strengths[i] += 1
        
        return strengths
    
    def compute_raw_fitness(self, population, fitnesses, cardinalities, strengths):
        """Raw fitness: sum of strengths of solutions that dominate this one"""
        n = len(population)
        raw_fitness = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and self.dominates(fitnesses[j], cardinalities[j],
                                            fitnesses[i], cardinalities[i]):
                    raw_fitness[i] += strengths[j]
        
        return raw_fitness
    
    def compute_density(self, population, fitnesses, cardinalities):
        """Density estimation using k-th nearest neighbor"""
        n = len(population)
        k = int(np.sqrt(n))  # k-th neighbor
        densities = np.zeros(n)
        
        for i in range(n):
            # Compute distances to all other solutions (in objective space)
            distances = []
            for j in range(n):
                if i != j:
                    # Euclidean distance in (fitness, cardinality) space
                    # Normalize to [0, 1] range
                    fit_diff = (fitnesses[i] - fitnesses[j]) / (max(fitnesses) - min(fitnesses) + 1e-9)
                    card_diff = (cardinalities[i] - cardinalities[j]) / (max(cardinalities) - min(cardinalities) + 1e-9)
                    dist = np.sqrt(fit_diff**2 + card_diff**2)
                    distances.append(dist)
            
            # Sort and get k-th distance
            distances.sort()
            k_dist = distances[min(k-1, len(distances)-1)]
            densities[i] = 1.0 / (k_dist + 2.0)  # Add 2 to avoid division by zero
        
        return densities
    
    def assign_fitness(self, population, fitnesses, cardinalities):
        """SPEA2 fitness assignment: raw fitness + density"""
        strengths = self.compute_strength(population, fitnesses, cardinalities)
        raw_fitness = self.compute_raw_fitness(population, fitnesses, cardinalities, strengths)
        densities = self.compute_density(population, fitnesses, cardinalities)
        
        # Final fitness = raw fitness + density
        # Lower is better (non-dominated have raw_fitness=0)
        final_fitness = raw_fitness + densities
        
        return final_fitness
    
    # ==================== ARCHIVE MANAGEMENT ====================
    
    def truncate_archive(self, archive, fitnesses, cardinalities, fitness_values):
        """Remove most crowded solutions from archive"""
        while len(archive) > self.archive_size:
            # Compute distances between all pairs
            n = len(archive)
            min_dist = float('inf')
            to_remove = 0
            
            for i in range(n):
                # Find closest neighbor
                closest_dist = float('inf')
                for j in range(n):
                    if i != j:
                        fit_diff = (fitnesses[i] - fitnesses[j]) / (max(fitnesses) - min(fitnesses) + 1e-9)
                        card_diff = (cardinalities[i] - cardinalities[j]) / (max(cardinalities) - min(cardinalities) + 1e-9)
                        dist = np.sqrt(fit_diff**2 + card_diff**2)
                        closest_dist = min(closest_dist, dist)
                
                # Mark solution with smallest distance to neighbor for removal
                if closest_dist < min_dist:
                    min_dist = closest_dist
                    to_remove = i
            
            # Remove the most crowded solution
            archive.pop(to_remove)
            fitnesses = np.delete(fitnesses, to_remove)
            cardinalities = np.delete(cardinalities, to_remove)
            fitness_values = np.delete(fitness_values, to_remove)
        
        return archive, fitnesses, cardinalities, fitness_values
    
    # ==================== MUTATION ====================
    
    def compute_power_law_distribution(self, n: int) -> np.ndarray:
        """Compute power-law distribution for mutation"""
        distribution = np.zeros(n + 1)
        C = sum(i ** (-self.beta) for i in range(1, n // 2 + 1))
        
        for i in range(1, min(n // 2 + 1, n + 1)):
            distribution[i] = (i ** (-self.beta)) / C
        
        distribution[0] = 0.0
        return distribution
    
    def mutate(self, individual: np.ndarray, n: int) -> np.ndarray:
        """Power-law mutation"""
        offspring = individual.copy()
        
        if self.power_law_distribution is None:
            self.power_law_distribution = self.compute_power_law_distribution(n)
        
        k = np.random.choice(n + 1, p=self.power_law_distribution)
        k = max(1, k)
        
        flip_indices = np.random.choice(n, size=k, replace=False)
        offspring[flip_indices] = 1 - offspring[flip_indices]
        
        return offspring
    
    # ==================== MAIN ALGORITHM ====================
    
    def __call__(self, func: ioh.problem.PBO):
        n = func.meta_data.n_variables
        
        # Pre-compute power-law distribution
        self.power_law_distribution = self.compute_power_law_distribution(n)
        
        # Initialize population (sparse initialization)
        population = []
        for _ in range(self.population_size):
            individual = np.zeros(n, dtype=int)
            num_ones = np.random.randint(max(1, n // 100), max(2, n // 50))  # 1-2%
            ones_positions = np.random.choice(n, size=num_ones, replace=False)
            individual[ones_positions] = 1
            population.append(individual)
        
        # Evaluate initial population
        fitnesses = np.array([func(ind.tolist()) for ind in population])
        cardinalities = np.array([np.sum(ind) for ind in population])
        
        # Initialize empty archive
        archive = []
        archive_fitnesses = np.array([])
        archive_cardinalities = np.array([])
        
        # Main loop
        while func.state.evaluations < self.budget:
            if func.state.optimum_found:
                break
            
            # Combine population and archive
            combined = population + archive
            combined_fits = np.concatenate([fitnesses, archive_fitnesses]) if len(archive) > 0 else fitnesses
            combined_cards = np.concatenate([cardinalities, archive_cardinalities]) if len(archive) > 0 else cardinalities
            
            # Filter feasible solutions only
            feasible_mask = combined_fits >= 0
            combined = [combined[i] for i in range(len(combined)) if feasible_mask[i]]
            combined_fits = combined_fits[feasible_mask]
            combined_cards = combined_cards[feasible_mask]
            
            if len(combined) == 0:
                break
            
            # Environmental selection: copy non-dominated to archive
            fitness_vals = self.assign_fitness(combined, combined_fits, combined_cards)
            non_dominated_indices = np.where(fitness_vals < 1.0)[0]  # raw_fitness=0 means non-dominated
            
            archive = [combined[i] for i in non_dominated_indices]
            archive_fitnesses = combined_fits[non_dominated_indices]
            archive_cardinalities = combined_cards[non_dominated_indices]
            archive_fitness_vals = fitness_vals[non_dominated_indices]
            
            # If archive too small, fill with best dominated
            if len(archive) < self.archive_size:
                dominated_indices = np.where(fitness_vals >= 1.0)[0]
                if len(dominated_indices) > 0:
                    sorted_dom = sorted(dominated_indices, key=lambda i: fitness_vals[i])
                    for idx in sorted_dom[:self.archive_size - len(archive)]:
                        archive.append(combined[idx])
                        archive_fitnesses = np.append(archive_fitnesses, combined_fits[idx])
                        archive_cardinalities = np.append(archive_cardinalities, combined_cards[idx])
                        archive_fitness_vals = np.append(archive_fitness_vals, fitness_vals[idx])
            
            # If archive too large, truncate
            if len(archive) > self.archive_size:
                archive, archive_fitnesses, archive_cardinalities, archive_fitness_vals = \
                    self.truncate_archive(archive, archive_fitnesses, archive_cardinalities, archive_fitness_vals)
            
            # Generate new population through binary tournament selection + mutation
            new_population = []
            new_fitnesses = []
            new_cardinalities = []
            
            mating_pool = archive if len(archive) > 0 else population
            mating_pool_fits = archive_fitness_vals if len(archive) > 0 else fitness_vals
            
            for _ in range(self.population_size):
                if func.state.evaluations >= self.budget:
                    break
                
                # Binary tournament selection
                idx1, idx2 = random.sample(range(len(mating_pool)), 2)
                parent = mating_pool[idx1] if mating_pool_fits[idx1] < mating_pool_fits[idx2] else mating_pool[idx2]
                
                # Mutate
                offspring = self.mutate(parent, n)
                
                # Evaluate
                offspring_fit = func(offspring.tolist())
                offspring_card = np.sum(offspring)
                
                new_population.append(offspring)
                new_fitnesses.append(offspring_fit)
                new_cardinalities.append(offspring_card)
            
            population = new_population
            fitnesses = np.array(new_fitnesses)
            cardinalities = np.array(new_cardinalities)