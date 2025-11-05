from .algorithm_interface import Algorithm
import ioh 
import numpy as np

import random
import matplotlib.pyplot as plot
import os

'''
Implementing the GSEMO algorithm to find the optimum solution of some predefined (bi-objective 
maximisation) problem 'func'. 

The algorithm uses a multi-objective fitness function as defined below for evaluation, and is 
uniformly constrained through parameter k. 
'''

class GSEMO(Algorithm):

    def __init__(self, budget: int):
        super().__init__(budget, name="GSEMO", algorithm_info="GSEMO algorithm implemented for maximisation with a uniform constraint.")
        self.budget = budget 


    def greedyInit(self, func, n, constraint):
        '''
        Helper function to 'greedily' initialise a binary string of length n. 
        (As is traditionally implemented for monotone submodular problems). 
        '''
        '''
        x = np.zeros(n, dtype = int)    # string to fill
        v = np.ones(n, dtype = bool)   # string of candidate elements

        evals_so_far = 0

        def limited_func(arr):
            ''''''
            Limits evaluations for initialisation when problem 
            instance has a large dimension. 
            ''''''

            nonlocal evals_so_far
            if evals_so_far >= self.budget/2:
                return 0
            evals_so_far += 1
            return func(arr)

        # Loop to evaluate each element's fitness; 
        # while v contains non-zero (1) elements AND x aligns with the constraint
        while (np.any(v) and np.sum(x) < constraint and evals_so_far < self.budget/2):  

            max_index = 0 
            max_gain = -np.inf
            for i in range(n):
                if (v[i] != 1 or x[i] == 1): 
                    continue
                
                temp = np.copy(x)   # string x w/ candidate being evaluated
                temp[i] = 1
                gain = (limited_func(temp) - limited_func(x))/(np.sum(temp) - np.sum(x))

                if gain > max_gain:
                    max_gain = gain
                    max_index = i
 
            # Add element to x 
            x[max_index] = 1

            # Remove evaluated candidate element 
            v[max_index] = 0

            evals_so_far += 1

        # Check if the highest value single element in v >> the generated x
        zer_arr = np.zeros(n, dtype = int) 
        fit_x = func(x)
        for i in range(n):
            temp = np.copy(zer_arr)
            temp[i] = 1
            if (func(temp) > fit_x):
                x = temp
                break 

        return x'''

        zer_arr = np.zeros(n, dtype = int) 
        x = np.copy(zer_arr)
        opt_fit = func(x)
        for i in range(n):
            temp = np.copy(zer_arr)
            temp[i] = 1
            if (func(temp) > opt_fit):
                x = temp 
        
        return x


    def multObjFit(self, x, x_fitn, y, y_fitn):
        '''
        Multi-objective fitness function that compares fitness and elements to verify 
        level of domination between solutions. 
        The function returns how x dominates y for some problem predefined func. 
        '''

        if (x_fitn >= y_fitn and np.sum(x) <= np.sum(y)):
            return "WEAK"
        if (x_fitn > y_fitn and np.sum(x) < np.sum(y)):
            return "STRICT"
        if (x_fitn < y_fitn and np.sum(x) > np.sum(y)):
            return "DOMINATED"
        else:
            return "INCOMPARABLE"

    def __call__(self, func: ioh.problem.GraphProblem):

        n = func.meta_data.n_variables
        mut_rate = 1/n 
        constraint_k = 0.001*n
        
        # Generate a bit string individual to initialise the pareto set
        x_init = np.array(np.random.randint(2, size = n))
        pareto = [(x_init, func(x_init))]

        '''# Generate a bit string individual to initialise the pareto set
        x_init = self.greedyInit(func, n, constraint_k)
        pareto = [(x_init, func(x_init))]   # use pairs to minimise no. of function evaluations '''

        # Loop of function evaluations: 
        while func.state.evaluations < self.budget:

            # Randomly select an individual from the pareto set
            x = random.choice(pareto)[0]

            # Mutate individual by flipping each bit with probability 1/n
            x_mut = np.copy(x)
            for j in range(n):
                if np.random.rand() < mut_rate:
                    x_mut[j] = 1 - x_mut[j]

            f_mut = func(x_mut)
            
            # Enforce uniform constraint (feasibility for maximum number of 1-bits k) by repairing individuals
            while (np.sum(x_mut) > constraint_k):
                one_indices = np.where(x_mut == 1)[0]
                x_mut[np.random.choice(one_indices)] = 0
            

            # Check if mutated individual is not strictly dominated by any individual in P (for a maximation problem)
            dominatingX = any(self.multObjFit(y, y_fit, x_mut, f_mut) == "STRICT" for y, y_fit in pareto)
            
            # If not strictly dominated...
            if not dominatingX: 
                # Delete all solutions that x weakly dominates
                pareto = [(y, y_fit) for y, y_fit in pareto if self.multObjFit(x_mut, f_mut, y, y_fit) != "WEAK"]  
                
                # Add to indivudual pareto set
                pareto.append((x_mut, f_mut))    


        # PLOTTING TRADE-OFFS BETWEEN OBJECTIVES: f(x) vs. number of 1-bit elements (as constrained by k)
        fit_dat = []
        k_dat = []
        for (x, f) in pareto:
            fit_dat.append(f)   # fitness of elements in pareto set
            k_dat.append(np.sum(x))     # number of 1-bit elements of each individual in set

        plot.scatter(k_dat, fit_dat)

        plot.xlabel("Number of Chosen Elements")
        plot.ylabel("Fitness Values")
        plot.grid(True)

        plot.savefig(f"GSEMO_{func.meta_data.problem_id}")
        
        # plot.savefig(os.path.join("", f"GSEMO {func}"))


# TO-DO: - Label latest run as 0.01n; evaluate which k does best
#        - Implement greedy initialisation of soln as in lectures
#        - Check effect -> change around k
#        - Trade-offs plot 