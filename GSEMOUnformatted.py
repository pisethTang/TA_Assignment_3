from ioh import get_problem, ProblemClass
from ioh import logger
import numpy as np
import random

'''
Implementing the GSEMO algorithm to find the optimum solution of some predefined (bi-objective 
maximisation) problem 'func'. 
The algorithm uses a multi-objective fitness function as defined below for evaluation, and is 
uniformly constrained through parameter k. 
'''

def multObjFit(x, x_fitn, y, y_fitn):
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

def GSEMO_algorithm(func, budget = None, k = 10): 

    n = func.meta_data.n_variables
    mut_rate = 1/n

    # Define budget (number of function evaluations) of each run: 10^4
    if budget is None:
        budget = int(pow(10, 4))

    # Print default optimum of the given problem 
    print("Optimum = ", func.optimum.y)
    
    # 30 independent runs for each algorithm on each problem. 
    for r in range(30):

        # Generate a bit string individual to initialise the pareto set
        x_init = np.array(np.random.randint(2, size = n))
        pareto = [(x_init, func(x_init))]

        # Loop of function evaluations: 
        for _ in range(budget):
            # Randomly select an individual from the pareto set
            x = random.choice(pareto)[0]
            
            # Mutate individual by flipping each bit with probability 1/n
            x_mut = np.copy(x)
            for j in range(n):
                if np.random.rand() < mut_rate:
                    x_mut[j] = 1 - x_mut[j]

            f_mut = func(x_mut)

            # Enforce uniform constraint: feasibility for maximum number of 1-bits k
            if (np.sum(x_mut) > k):
                continue

            # Check if mutated individual is not strictly dominated by any individual in P (for a maximation problem)
            dominatingX = any(multObjFit(y, y_fit, x_mut, f_mut) == "STRICT" for y, y_fit in pareto)
            
            # If not strictly dominated...
            if not dominatingX: 
                # Delete all solutions that x weakly dominates
                pareto = [(y, y_fit) for y, y_fit in pareto if multObjFit(x_mut, f_mut, y, y_fit) != "WEAK"]  
                
                # Add to indivudual pareto set
                pareto.append((x_mut, f_mut))    

            '''
            dominates = False
            for j in range(len(pareto)):
                if (multObjFit(func, x_mut, pareto[j]) == "DOMINATED" or "WEAK"):
                    dominates = False
                    break
                dominates = True

            # If individual not strictly dominated...
            if (dominates):
                pareto.append(x_mut)    # Add to pareto set

                for j in range(len(pareto) - 1):     # Delete all solutions that x weakly dominates
                    if (multObjFit(func, x_mut, pareto[j]) == "WEAK" or "STRONG"):
                        pareto.remove(pareto[j])
            '''
        
        # Reset function
        func.reset() 

        print(f"Run {r + 1} of 30 complete!")

    # Return pareto set
    return pareto

# Declaration of problems to be tested:
# F2100 = MaxCoverage; F2200 = MaxInfluence; F2300 = PackWhileTravel 
# dimension = number of variables
cov = get_problem(fid = 2100, problem_class = ProblemClass.GRAPH)
infl = get_problem(fid = 2200, problem_class = ProblemClass.GRAPH)
pack = get_problem(fid = 2300, problem_class = ProblemClass.GRAPH)


# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="GSEMO", 
    algorithm_info="test of IOHexperimenter in python")


print("Optimising F2100: ")
cov.attach_logger(l)
GSEMO_algorithm(cov)

print("Optimising F2200: ")
infl.attach_logger(l)
GSEMO_algorithm(infl)

print("Optimising F2300: ")
pack.attach_logger(l)
GSEMO_algorithm(pack)

# This statement is necessary in case data is not flushed yet.
del l