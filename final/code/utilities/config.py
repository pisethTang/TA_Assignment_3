# import necessary module(s)
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
BUDGET = 100000   # maximum number of function evaluations per run (or number of iterations or generations for GAs)
DIMENSION = 100   # problem dimension/size (e.g., number of bits for OneMax and LeadingOnes)
REPETITIONS = 30  # number of independent repetitions or runs for each problem

PROBLEM_IDS = [ # list of problems, identified by the following IDs, to be run in our experiment in `main.py`
    # 2100, # MaxCoverage Problem
    # 2101,
    # 2102,
    # 2103,

    2200, # MaxInfluence Problem
    2201,
    2202,
    2203,


    # ---- Not required for exercises 3 and 4 ----
    # 2300, # PackWhileTravel Problem
    # 2301,
    # 2302
]
# PROBLEMS_TYPE = ioh.ProblemClass.PBO  # Pseudo-Boolean Optimization problems
PROBLEMS_TYPE = ioh.ProblemClass.GRAPH  # Graph problems


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
    SingleObjectiveEA(budget=BUDGET, 
                    population_size=POPULATION_SIZES[1],
                    beta=1.5,
                    tournament_size=6,
                    # patience_factor=0.1,
                    ),
    # SingleObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[2],
    #                 beta=1.5,
    #                 tournament_size=8,
    #                 ),
    # ========================================================================




    # MultiObjectiveEA's runs (uncomment to test)
    # MultiObjectiveEA(budget=BUDGET, 
    #                 population_size=POPULATION_SIZES[0],
    #                 archive_size=10,
    #                 beta=1.5),

    # MultiObjectiveEA(budget=BUDGET, population_size=POPULATION_SIZES[1]),
    # MultiObjectiveEA(budget=BUDGET, population_size=POPULATION_SIZES[2]),
]