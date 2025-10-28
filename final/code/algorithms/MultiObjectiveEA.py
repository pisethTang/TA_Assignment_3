import ioh 
from .algorithm_interface import Algorithm
import numpy as np 



class MultiObjectiveEA(Algorithm):
    """Population-based Multi Objective Evolutionary Algorithm implementation."""
    def __init__(self, budget: int,
                name: str = "MultiObjectiveEA",
                algorithm_info: str = "Population-based Multi Objective Evolutionary Algorithm implementation.",
                population_size: int = 10,
                ):
        super().__init__(budget, name=name, algorithm_info=algorithm_info)
        self.population_size = population_size
