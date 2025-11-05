<<<<<<< HEAD
# import necessary module(s)
=======
from algorithms import RandomSearch, RandomizedLocalSearch, MaxMinAS, DesignedGA, ACO, MaxMinASStar, OnePlusOneEA, SOP_EA
import math
>>>>>>> Theories
import ioh

from algorithms import (
    # For exercise 1
    RandomSearch,
    OnePlusOneEA,
    RandomizedLocalSearch,
    DesignedGA,


    # For exercise 2
    GSEMO_Seth,

    # For exercises 3 and 4
    SingleObjectiveEA,
    MultiObjectiveEA,
)








# configuration parameters for the experiments
BUDGET_1 = 10000   # maximum number of function evaluations per run (or number of iterations or generations for GAs)
BUDGET_2 = 100000  # maximum number of function evaluations per run (for exercises 3 and 4)
BUDGET = BUDGET_1  # default budget to use


DIMENSION = 100   # problem dimension/size (e.g., number of bits for OneMax and LeadingOnes)
REPETITIONS = 30  # number of independent repetitions or runs for each problem

PROBLEM_IDS = [ # list of problems, identified by the following IDs, to be run in our experiment in `main.py`
    # 2100, # MaxCoverage Problem
    # 2101,
    # 2102,
    # 2103,

    # 2200, # MaxInfluence Problem
    # 2201,
    # 2202,
    2203,


    # ---- Not required for exercises 3 and 4 ----
    # 2300, # PackWhileTravel Problem
    # 2301,
    # 2302
]
# PROBLEMS_TYPE = ioh.ProblemClass.PBO  # Pseudo-Boolean Optimization problems
PROBLEMS_TYPE = ioh.ProblemClass.GRAPH  # Graph problems
B = 10  # cost constraint

POPULATION_SIZES = [10, 
                    20, 
                    50]  # Different population sizes to experiment with for population-based algorithms
# a list of algorithm instances to run 
ALGORITHMS = [
    
    # RandomSearch(budget=BUDGET),
    # OnePlusOneEA(budget=BUDGET),
    # RandomizedLocalSearch(budget=BUDGET),
    # DesignedGA(budget=BUDGET, population_size=44, mutation_rate=0.01),
    # ACO(budget=BUDGET)
    # GSEMO_Seth(budget=BUDGET) 
    
    
    # SingleObjectiveEA's runs (uncomment to test)
    # ================================== USE PARAMETERS BELOW ======================================
    # SingleObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[0],
    #                 beta=1.5,
    #                 tournament_size=3,
    #                 ),
    # SingleObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[1],
    #                 beta=1.5,
    #                 tournament_size=6,
    #                 # patience_factor=0.1,
    #                 ),
    # SingleObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[2],
    #                 beta=1.5,
    #                 tournament_size=8,
    #                 ),
    
    # ========================================================================




    # MultiObjectiveEA's runs (uncomment to test)
    MultiObjectiveEA(
        budget=BUDGET,
        population_size=POPULATION_SIZES[0],
        K_Elites=5,           # Use top 50% as parents (moderate selection)
        mutation_prob=None    # Default 1/n (adaptive to problem size)
    ),
    # MultiObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[1],
    #                 heavy_mix=0.15,
    #                 k_max=10,
    #                 parent_selection="crowded",
    #                 ),
    # MultiObjectiveEA(
    #     budget=BUDGET,
    #     population_size=10,
    #     mutation_mode="hybrid",    # 'uniform' | 'heavy' | 'hybrid'
    #     heavy_mix=0.10,            # 10% heavy-tailed, 90% uniform
    #     k_max=10,                  # cap heavy flips
    #     beta=1.5,                  # required by your course
    #     mutation_prob=None,        # uniform side defaults to 1/n
    #     parent_selection="crowded",# or "elites"
    #     # If you prefer elites for pop 10:
    #     # parent_selection="elites", K_Elites=5,
    #     B=B,
    #     name="NSGAII-hybrid-10"
    # ),    
    
    
    # pop = 20
    # MultiObjectiveEA(
    #     budget=BUDGET,
    #     population_size=20,
    #     mutation_mode="hybrid",
    #     heavy_mix=0.15,
    #     k_max=10,
    #     beta=1.5,
    #     mutation_prob=None,
    #     parent_selection="crowded",
    #     # or elites: K_Elites=10
    #     B=B,
    #     name="NSGAII-20"
    # ),


    # pop = 50
    # MultiObjectiveEA(
    #     budget=BUDGET,
    #     population_size=50,
    #     mutation_mode="hybrid",
    #     heavy_mix=0.15,
    #     k_max=10,
    #     beta=1.5,
    #     mutation_prob=None,
    #     parent_selection="crowded",
    #     # or elites: K_Elites=25
    #     B=B,
    #     name="NSGAII-hybrid-50"
    # ),
    # MultiObjectiveEA(budget=BUDGET, population_size=POPULATION_SIZES[2]),
]