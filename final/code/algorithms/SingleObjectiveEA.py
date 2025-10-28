import ioh 
from .algorithm_interface import Algorithm
import numpy as np 



class SingleObjectiveEA(Algorithm):
    """Population-based Single Objective Evolutionary Algorithm implementation."""
    def __init__(self, budget: int,
                name: str = "SingleObjectiveEA",
                algorithm_info: str = "Single Objective Evolutionary Algorithm implementation.",
                population_size: int = 10,
                ):
        super().__init__(budget, name=name, algorithm_info=algorithm_info)
        self.population_size = population_size
