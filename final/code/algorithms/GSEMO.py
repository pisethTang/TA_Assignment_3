from .algorithm_interface import Algorithm
import ioh 
import numpy as np

import random
import matplotlib.pyplot as plot
import os

'''
Implementing the GSEMO algorithm to find the optimum solution of some predefined (bi-objective 
maximisation) problem 'func'. 

The algorithm uses a multi-objective fitness function as defined below for evaluation. 
'''

class GSEMO(Algorithm):

    def __init__(self, budget: int):
        super().__init__(budget, name="GSEMO", algorithm_info="GSEMO algorithm implemented for maximisation with a uniform constraint.")
        self.budget = budget 

    def multObjFit(self, x, x_fitn, y, y_fitn):
        '''
        Multi-objective fitness function that compares fitness and elements to verify 
        level of domination between solutions. 
        The function returns how x dominates y for some problem predefined func. 
        '''

        if (x_fitn > y_fitn and np.sum(x) <= np.sum(y)):
            return "STRICT"
        if (x_fitn >= y_fitn and np.sum(x) <= np.sum(y)):
            return "WEAK"
        if (x_fitn < y_fitn and np.sum(x) > np.sum(y)):
            return "DOMINATED"
        else:
            return "INCOMPARABLE"

    def __call__(self, func: ioh.problem.GraphProblem):

        n = func.meta_data.n_variables
        mut_rate = 1/n 
        
        # Generate a bit string individual to initialise the pareto set
        x_init = np.array(np.random.randint(2, size = n))
        pareto = [(x_init, func(x_init))]

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


            # Optimisation step: 
            # Ensure fitness is positive; if not, repair by flipping one bits
            if (f_mut < 0): 
                # Option to repair solution by flipping 1 bits 
                one_indices = np.where(x_mut == 1)[0]
                for i in one_indices:
                    x_mut[i] = 0
                    f_new = func(x_mut)
                    if f_new >= 0:
                        f_mut = f_new
                        break    
            
            '''
            # Optimisation step? 
            # Enforce uniform constraint (feasibility for maximum number of 1-bits k) by repairing individuals
            
            while (np.sum(x_mut) > constraint_k):
                one_indices = np.where(x_mut == 1)[0]
                x_mut[np.random.choice(one_indices)] = 0
            '''

            # Check if mutated individual is not strictly dominated by any individual in P (for a maximation problem)
            dominatingX = any(self.multObjFit(y, y_fit, x_mut, f_mut) == "STRICT" for y, y_fit in pareto)
            
            # If not strictly dominated...
            if not dominatingX: 

                
                # Delete all solutions that x weakly dominates
                pareto = [(y, y_fit) for y, y_fit in pareto if self.multObjFit(x_mut, f_mut, y, y_fit) != "WEAK"]  
                
                '''
                # Alternative deletion that optimises; inapplicable for the given pseudocode 
                pareto = [(y, y_fit) for y, y_fit in pareto if self.multObjFit(x_mut, f_mut, y, y_fit) != "STRICT"]  
                '''

                # Add to indivudual pareto set
                pareto.append((x_mut, f_mut))    

        
        # PLOTTING TRADE-OFFS BETWEEN OBJECTIVES: f(x) vs. number of 1-bit elements (as constrained by k)
        dir = "trade-off plots"
        os.makedirs(dir, exist_ok=True)
        out_path = os.path.join(dir, f"GSEMO_F{func.meta_data.problem_id}")
       
        if not os.path.exists(out_path):  
            fit_dat = []
            k_dat = []
            for (x, f) in pareto:
                fit_dat.append(f)   # fitness of elements in pareto set
                k_dat.append(np.sum(x))     # number of 1-bit elements of each individual in set

            plot.figure()
            plot.scatter(k_dat, fit_dat)
            plot.xlabel("Number of Chosen Elements (1-bits)")
            plot.ylabel("Fitness Values")
            plot.grid(True)
            plot.title(f"GSEMO Trade-Offs for F{func.meta_data.problem_id}")
            plot.savefig(out_path)
            plot.close()
        

 