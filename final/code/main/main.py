import sys
from pathlib import Path
import time




# Add the parent directory (code/) to the Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities import config
from utilities.utilities import ensure_dir
import ioh

def get_unique_folder_name(base_dir: Path, algorithm_name: str, prefix: str = "ioh-data") -> str:
    """
    Produce a folder name of the form "{prefix}-{algorithm_name}" that does not
    clash with existing folders inside base_dir. If there's a clash, append -1, -2, ...
    Returns the folder name (string) to be passed as ioh.Experiment(..., folder_name=...).
    """
    base = f"{prefix}-{algorithm_name}"
    candidate = base_dir / base
    if not candidate.exists():
        return base
    # Otherwise find next available index
    idx = 1
    while True:
        name = f"{base}-{idx}"
        if not (base_dir / name).exists():
            return name
        idx += 1


def main():
    """
    Main execution function to **run** all configured algorithms on specified problems.
    """

    print("Starting experiments...")

    # ensures the doc/data directory exists and output there
    # Path is relative to the main.py location: main/ -> code/ -> final/ -> doc/data/
    out_base = ensure_dir(Path(__file__).parent.parent.parent / "doc" / "data")


    # based on the number of algorithms
    # create an array of elapsed times, on for each experiment
    elapsed_times = []

    for algorithm in config.ALGORITHMS:
        print(f"=========== Running experiments for algorithm: {algorithm.name} on problems {config.PROBLEM_IDS} pop_size={algorithm.population_size} with a budget={config.BUDGET} ========== ")
        # start time - use perf_counter for accurate wall-clock time measurement
        start_time = time.perf_counter()

        unique_folder = get_unique_folder_name(out_base, algorithm.name)
        # create a new experiment for the current algorithm 
        experiment = ioh.Experiment(
            algorithm=algorithm,
            algorithm_name=f"{algorithm.name}-{config.BUDGET}-pop{algorithm.population_size}",
            algorithm_info=algorithm.algorithm_info,
            fids = config.PROBLEM_IDS,
            iids = [1], 
            dims=[config.DIMENSION],
            reps=config.REPETITIONS,
            problem_class=config.PROBLEMS_TYPE,  # Use the configured problem class # type: ignore
            old_logger=False,  # type: ignore
            output_directory=str(out_base),
            folder_name=f"ioh-data-exercise3-MOEA-{algorithm.name}", # for population-based algorithms (single objective and multi-objective), technically for our designed GA as well.
            merge_output=False, # whether to merge with existing data (of same name) or create new files
            zip_output=True, 
        )


        experiment.run()
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        elapsed_times.append(elapsed)
        print(f"Elapsed time for algorithm {algorithm.name}: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes) after {config.REPETITIONS} runs on {len(config.PROBLEM_IDS)} problems.")
        print(f"=========== Completed experiments for algorithm: {algorithm.name} ========== ")

    
    print("All experiments completed.")
    print(f"Results are saved in the '{out_base}' directory.")
    print("Summary of elapsed times for each algorithm:")
    # convert elapsed time to minutes and print
    for alg, t in zip(config.ALGORITHMS, elapsed_times):
        print(f"Total time for {alg.name}: {t:.2f} seconds, which is around ({t/60:.2f} minutes)")


if __name__ == "__main__":
    main()



