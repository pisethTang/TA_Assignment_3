import sys
from pathlib import Path
import matplotlib.pyplot as plt
import ioh

# Add the parent directory ('final/code') to the Python path
current_dir = Path(__file__).parent
code_dir = current_dir.parent 
sys.path.insert(0, str(code_dir))

# Import config which defines ALGORITHMS and other constants
from utilities import config 

# Import utility to create directories
from utilities.utilities import ensure_dir 




def generate_plots(target_budget: int, output_suffix: str):
    """
    Generates trade-off plots for multi-objective algorithms defined in config.py 
    for specified instances and budget.
    """
    
    # --- Base output directory for all plots ---
    base_plot_output_dir = code_dir.parent.parent / "doc" / "plots"
    ensure_dir(base_plot_output_dir)

    print(f"\n--- Generating Trade-off Plots for Budget: {target_budget} ---")

    # --- Loop through algorithms defined in config.py ---
    for algorithm_instance in config.ALGORITHMS:
        
        print(f"\n--- Processing Algorithm: {algorithm_instance.name} ---")
        
        # --- Create a subdirectory for this algorithm's plots ---
        algo_plot_dir = base_plot_output_dir / f"{algorithm_instance.name}_{output_suffix}"
        ensure_dir(algo_plot_dir)

        # Define which problem instances to plot (adjust if needed for specific exercises)
        plot_problem_ids = config.PROBLEM_IDS

        for problem_id in plot_problem_ids:
            print(f"  Processing Problem ID: {problem_id}...")
            try:
                # --- Load Problem Instance ---
                problem = ioh.get_problem(
                    fid=problem_id, 
                    instance=1, # First run
                    dimension=config.DIMENSION, 
                    problem_class=config.PROBLEMS_TYPE
                )

                # --- Instantiate a fresh copy for the run (important!) ---
                # Re-create the instance to ensure budget/state is reset
                # This assumes your classes in config.ALGORITHMS are types or have params stored
                algo_runner = type(algorithm_instance)(budget=target_budget) # Re-instantiate

                # --- Run Algorithm ONCE ---
                problem.reset()  # Reset problem state before running the current algorithm on a problem.

                # the final population is a list of (individual, fitness) tuples
                final_population = algo_runner.run_optimization_loop(problem) 

                if not final_population:
                    print(f"  Warning: No solutions found for problem {problem_id}. Skipping plot.")
                    continue

                scores = [fit[0] for _, fit in final_population]
                costs = [fit[1] for _, fit in final_population]

                plt.figure(figsize=(8, 6))
                plt.scatter(scores, costs, alpha=0.7, s=10) # Smaller points might look better
                
                plt.title(f"{algorithm_instance.name} Trade-offs (First Run)\nProblem: {problem.meta_data.name}, Budget: {target_budget}")
                plt.xlabel("Objective 1: Score (Maximize)")
                plt.ylabel("Objective 2: Cost (Minimize)")
                plt.grid(True)
                
                plot_filename = algo_plot_dir / f"tradeoff_{problem.meta_data.name}_b{target_budget}.png"
                plt.savefig(plot_filename, dpi=150) # Increase dpi for better resolution
                plt.close() 
                print(f"  Saved plot: {plot_filename.name}")

            except Exception as e:
                print(f"  Error processing problem {problem_id} for {algorithm_instance.name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Generate plots for Exercise 2 budget (10k)
    generate_plots(target_budget=config.BUDGET, output_suffix="ex2_10k")
    
    # Generate plots for Exercise 4 budget (100k)
    # generate_plots(target_budget=100000, output_suffix="ex4_100k") 

    print("\n--- Plot generation complete ---")