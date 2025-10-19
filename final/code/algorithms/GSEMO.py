import ioh 
from .algorithm_interface import Algorithm
import numpy as np



class GSEMO(Algorithm):
    '''
    Global Simple Evolutionary Multi-Objective Optimizer (GSEMO) algorithm implementation.
    '''
    def __init__(self, budget: int,
                 name: str = "GSEMO",
                 algorithm_info: str = "Global Simple Evolutionary Multi-Objective Optimizer (GSEMO) algorithm implementation."):
        super().__init__(budget, name=name, algorithm_info=algorithm_info)



    def evaluate_individual_multiobjective_fitness(self, individual: np.ndarray, func) -> np.ndarray:
        '''
            Evaluates the multi-objective fitness of a given individual using the provided function.
        '''

        # objective 1: score (to be maximized)
        score = func(individual.tolist())
        
        # objective 2: cost (to be minimized)
        # since the cost is assumed to be uniform (basically constant), that is,
        # the cost of each edge is 1, then the cost of an individual is simply the number of 1s in its bitstring representation 
        # i.e. cost(individual) = sum(individual[i] * cost(edge_i)) = sum(individual[i] * 1) = sum(individual)

        cost = np.sum(individual)  # number of 1s in the bitstring

        return (score, cost)

    def GSEMO_algorithm(self, func, n: int) -> np.ndarray:
        '''
        Main GSEMO algorithm implementation.
        '''

        # Initialise population with a single individual (all zeros)
        population = [np.zeros(n, dtype=int)]

        while func.state.evaluations < self.budget:

            # Select a parent uniformly at random from the population
            parent = population[np.random.randint(len(population))]

            # Create an offspring by flipping a random bit
            offspring = parent.copy()
            mutation_index = np.random.randint(n)
            offspring[mutation_index] = 1 - offspring[mutation_index]  # flip the bit

            # Evaluate the multi-objective fitness of the offspring
            offspring_fitness = self.evaluate_individual_multiobjective_fitness(offspring, func)

            # Check if the offspring is dominated by any individual in the population
            dominated = False
            non_dominated_indices = []
            for i, individual in enumerate(population):
                individual_fitness = self.evaluate_individual_multiobjective_fitness(individual, func)

                if (individual_fitness[0] >= offspring_fitness[0] and individual_fitness[1] <= offspring_fitness[1]) and (individual_fitness != offspring_fitness):
                    dominated = True
                    break
                elif (offspring_fitness[0] >= individual_fitness[0] and offspring_fitness[1] <= individual_fitness[1]) and (offspring_fitness != individual_fitness):
                    continue  # offspring dominates this individual
                else:
                    non_dominated_indices.append(i)

            # If the offspring is not dominated, add it to the population
            if not dominated:
                population.append(offspring)

            # Keep only non-dominated individuals in the population
            population = [population[i] for i in non_dominated_indices]

        # Return the best individual found
        return population[np.argmax([self.evaluate_individual_multiobjective_fitness(ind, func)[0] for ind in population])]