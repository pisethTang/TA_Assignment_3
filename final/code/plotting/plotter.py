import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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
    base_plot_output_dir = code_dir.parent.parent / "doc" / "Tradeoff_plots"
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
                # final_population = algo_runner.run_optimization_loop(problem) 
                final_population = algo_runner(problem)  # call __call__

                # Normalize fitness values so every entry becomes (score, cost)
                norm_pop = []
                for indiv, fit in final_population:
                    # case: fit is scalar (float/int)
                    if isinstance(fit, (float, int, np.floating, np.integer)):
                        score = float(fit)
                        cost  = int(np.sum(indiv))            # default 2nd objective = #1-bits
                    else:
                        arr = np.asarray(fit)
                        if arr.size >= 2:
                            score, cost = float(arr[0]), float(arr[1])
                        elif arr.size == 1:
                            score, cost = float(arr[0]), int(np.sum(indiv))
                        else:
                            # skip malformed entry
                            continue
                    norm_pop.append((indiv, (score, cost)))

                if not norm_pop:
                    print(f"  Warning: No plottable fitness values for problem {problem_id}. Skipping plot.")
                    continue

                # ---------- FILTERING and PLOTTING (paste here, replacing current scores/costs + plotting) ----------
                # Keep only solutions that satisfy cost <= 10 (your requirement)
                cost_limit = 10
                filtered = [(ind, f) for ind, f in norm_pop if f[1] <= cost_limit]
                print(f"  Info: {len(norm_pop)} total points, {len(filtered)} points with cost <= {cost_limit}")

                if not filtered:
                    print(f"  No points with cost <= {cost_limit} for problem {problem_id}. Using all points instead.")
                    filtered = norm_pop

                # Unpack into arrays
                scores = np.asarray([s for _, (s, _) in filtered])
                costs  = np.asarray([c for _, (_, c) in filtered])

                # OPTIONAL: further remove fitness outliers using IQR (set to False to skip)
                use_iqr_filter = True
                iqr_multiplier = 1.5  # common choice; increase to be less aggressive

                if use_iqr_filter and scores.size > 0:
                    q1, q3 = np.percentile(scores, [25, 75])
                    iqr = q3 - q1
                    low_cut = q1 - iqr_multiplier * iqr
                    high_cut = q3 + iqr_multiplier * iqr
                    iqr_mask = (scores >= low_cut) & (scores <= high_cut)
                    kept = np.count_nonzero(iqr_mask)
                    print(f"  IQR filtering: kept {kept}/{scores.size} points (range [{low_cut:.3g}, {high_cut:.3g}])")
                    # apply mask
                    scores = scores[iqr_mask]
                    costs  = costs[iqr_mask]

                if scores.size == 0:
                    print(f"  Warning: All points removed by outlier filter for problem {problem_id}. Skipping plot.")
                    continue

                # ------ HEXBIN (filtered) ------
                plt.figure(figsize=(8, 6))
                plt.hexbin(scores, costs, gridsize=50, mincnt=1)
                plt.colorbar(label="Count")
                plt.xlabel("Objective 1: Score (Maximize)")
                plt.ylabel("Number of 1-bits (Chosen Elements)")
                plt.title(f"{algorithm_instance.name} Trade-offs (Filtered)\nProblem: {problem.meta_data.name}, Budget: {target_budget}")
                if costs.max() - costs.min() <= 50:
                    plt.yticks(np.arange(int(costs.min()), int(costs.max())+1))
                plt.tight_layout()
                plot_filename = algo_plot_dir / f"hexbin_filtered_{problem.meta_data.name}_b{target_budget}.png"
                plt.savefig(plot_filename, dpi=150)
                plt.close()
                print(f"  Saved hexbin (filtered): {plot_filename.name}")

                # ------ CIRCLE-BIN (filtered) ------
                # build 2D histogram counts, then draw circles at bin centers sized by count
                x_bins = 50
                y_range = int(max(1, costs.max() - costs.min() + 1))
                y_bins = min(max(10, y_range), 200)

                H, xedges, yedges = np.histogram2d(scores, costs, bins=[x_bins, y_bins])

                xcenters = (xedges[:-1] + xedges[1:]) / 2.0
                ycenters = (yedges[:-1] + yedges[1:]) / 2.0

                X, Y = np.meshgrid(xcenters, ycenters, indexing="xy")
                Xf = X.ravel()
                Yf = Y.ravel()
                counts = H.T.ravel()   # transpose so counts align with X,Y

                mincnt = 1
                mask = counts >= mincnt
                Xf = Xf[mask]; Yf = Yf[mask]; counts = counts[mask]

                max_size = 600
                sizes = (counts / counts.max()) * max_size
                alpha = 0.8

                plt.figure(figsize=(8, 6))
                sc = plt.scatter(Xf, Yf, s=sizes, c=counts, cmap="viridis", alpha=alpha, marker="o", edgecolors="none")
                plt.colorbar(sc, label="Count")
                plt.xlabel("Objective 1: Score (Maximize)")
                plt.ylabel("Number of 1-bits (Chosen Elements)")
                plt.title(f"{algorithm_instance.name} Trade-offs (Filtered, Circles)\nProblem: {problem.meta_data.name}, Budget: {target_budget}")
                if costs.max() - costs.min() <= 50:
                    plt.yticks(np.arange(int(costs.min()), int(costs.max())+1))
                plt.tight_layout()
                plot_filename = algo_plot_dir / f"circles_filtered_{problem.meta_data.name}_b{target_budget}.png"
                plt.savefig(plot_filename, dpi=150)
                plt.close()
                print(f"  Saved circle-binned (filtered): {plot_filename.name}")
                # ---------- end of block ----------


            except Exception as e:
                print(f"  Error processing problem {problem_id} for {algorithm_instance.name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Generate plots for Exercise 2 budget (10k)
    generate_plots(target_budget=config.BUDGET, output_suffix="ex2_10k")
    
    # Generate plots for Exercise 4 budget (100k)
    # generate_plots(target_budget=100000, output_suffix="ex4_100k") 

    print("\n--- Plot generation complete ---")