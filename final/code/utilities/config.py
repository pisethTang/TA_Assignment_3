from algorithms import RandomSearch, RandomizedLocalSearch, MaxMinAS, DesignedGA, ACO, MaxMinASStar, OnePlusOneEA, GSEMO
import math
import ioh


# configuration parameters for the experim  ents
BUDGET = 10000   # maximum number of function evaluations per run (or number of iterations or generations for GAs)
DIMENSION = 450   # problem dimension/size (e.g., number of bits for OneMax and LeadingOnes)
# REPETITIONS = 10  # number of independent repetitions or runs for each problem
REPETITIONS = 1  # number of independent repetitions or runs for each problem

PROBLEM_IDS = [ # list of problems, identified by the following IDs, to be run in our experiment in `main.py`
    # 2100, # MaxCoverage Problem
    # 2101,
    # 2102,
    # 2103,
    # 2200, # MaxInfluence Problem
    # 2201,
    # 2202,
    2203,
    # 2300, # PackWhileTravel Problem
    # 2301,
    # 2302
]
# PROBLEMS_TYPE = ioh.ProblemClass.PBO  # Pseudo-Boolean Optimization problems
PROBLEMS_TYPE = ioh.ProblemClass.GRAPH  # Graph problems

# a list of algorithm instances to run 
ALGORITHMS = [
    # MaxMinASStar(budget=BUDGET, evaporate_rate=1),
    # MaxMinASStar(budget=BUDGET, evaporate_rate=1/math.sqrt(DIMENSION)),
    # MaxMinASStar(budget=BUDGET, evaporate_rate=1/DIMENSION),
    # MaxMinAS(budget=BUDGET, evaporate_rate=1),
    # MaxMinAS(budget=BUDGET, evaporate_rate=1/math.sqrt(DIMENSION)),
    # MaxMinAS(budget=BUDGET, evaporate_rate=1/DIMENSION),
    # RandomSearch(budget=BUDGET),
    # OnePlusOneEA(budget=BUDGET),
    # RandomizedLocalSearch(budget=BUDGET),
    # DesignedGA(budget=BUDGET, population_size=44, mutation_rate=0.01),
    # ACO(budget=BUDGET)
    GSEMO(budget=BUDGET)
]