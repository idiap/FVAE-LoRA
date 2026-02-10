# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import re
import numpy as np
from collections import defaultdict


def extract_metric_from_file(file_path: Path, metric_name: str) -> float | None:
    """
    Extracts a specific metric (like test_acc) from a log file.
    Searches for a line starting with the metric_name (possibly indented),
    followed by whitespace, then a number.
    Example line in log: "    test_acc            0.5446808338165283"

    Args:
        file_path (Path): Path to the log file.
        metric_name (str): The name of the metric to extract (e.g., "test_acc").

    Returns:
        float | None: The extracted metric value as a float, or None if not found or an error occurs.
    """
    pattern = rf"^\s*{re.escape(metric_name)}\s+([\d\.]+)"
    found_metric_value = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    found_metric_value = float(match.group(1))
                    print(
                        f"    Found {metric_name}: {found_metric_value:.4f} in '{file_path.name}'"
                    )
                    break
    except FileNotFoundError:
        print(f"    Error: Log file not found {file_path}")
        return None
    except Exception as e:
        print(f"    Error reading or parsing file {file_path}: {e}")
        return None

    if found_metric_value is None:
        print(f"    {metric_name} not found in '{file_path.name}'")
    return found_metric_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl results from experiment logs, grouped by experiment configuration name found across seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp-base",  # Renamed for clarity
        type=Path,
        required=True,
        help="Base directory containing subdirectories for each seed (e.g., '1', '2', '42'). "
        "Each seed subdirectory is expected to contain further subdirectories for each "
        "experiment configuration name. "
        "Example: --exp-base ./my_runs/",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[1, 2, 42],
        help="List of seed numbers (directory names) to process under --exp-base.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        choices=[1, 2],
        help="Maximum depth of experiment configuration directories to search under each seed directory.",
    )
    parser.add_argument(
        "--metric-string",
        type=str,
        default="test_acc",
        help="The metric string to search for in the log files (e.g., 'test_acc', 'test_loss').",
    )
    parser.add_argument(
        "--log-file-pattern",
        type=str,
        default="train.*.out",
        help="Glob pattern for log files within each experiment configuration directory (e.g., 'train.*.out', 'output.log').",
    )

    args = parser.parse_args()
    print(f"Running with arguments: {args}\n")

    # This dictionary will store data like:
    # {
    #   "exp_config_name_A": {1: 0.90, 2: 0.91, 42: 0.89},
    #   "exp_config_name_B": {1: 0.85, 42: 0.86}, # Seed 2 might not have exp_B
    #   ...
    # }
    # Using defaultdict to easily append to lists or create new seed dicts
    results_by_exp_name = defaultdict(dict)  # exp_name -> {seed: value}

    if not args.exp_base.is_dir():
        print(
            f"Error: Experiment base directory '{args.exp_base}' does not exist or is not a directory."
        )
        exit(1)

    # Loop through each specified seed directory
    for seed_val in args.seed:
        seed_str = str(seed_val)
        seed_dir_path = args.exp_base / seed_str

        print(
            f"\nProcessing Seed Directory: {seed_dir_path.relative_to(args.exp_base.parent)}"
        )

        if not seed_dir_path.is_dir():
            print(
                f"  Seed directory '{seed_dir_path.relative_to(args.exp_base.parent)}' not found. Skipping seed {seed_val}."
            )
            continue

        # Discover experiment configuration directories within this seed directory
        if args.max_depth == 1:
            exp_config_dirs_in_seed = [seed_dir_path]
        else:  # max_depth == 2
            exp_config_dirs_in_seed = sorted(
                [d for d in seed_dir_path.iterdir() if d.is_dir()]
            )

        if not exp_config_dirs_in_seed:
            print(
                f"  No experiment configuration subdirectories found in '{seed_dir_path.relative_to(args.exp_base.parent)}'."
            )
            continue

        for exp_config_path in exp_config_dirs_in_seed:
            if args.max_depth == 1:
                exp_config_name = exp_config_path.parent.name
            else:  # max_depth == 2
                exp_config_name = exp_config_path.name
            print(
                f"  Found Experiment Configuration: '{exp_config_name}' under seed {seed_val}"
            )

            # Find log files in that experiment configuration directory
            log_files = sorted(list(exp_config_path.glob(args.log_file_pattern)))

            if not log_files:
                print(
                    f"    No log files matching '{args.log_file_pattern}' found in '{exp_config_path.relative_to(args.exp_base.parent)}'."
                )
                results_by_exp_name[exp_config_name][
                    seed_val
                ] = None  # Mark as no log file
                continue

            selected_log_file: Path
            if len(log_files) > 1:
                selected_log_file = max(log_files, key=lambda f: f.stat().st_mtime)
                print(
                    f"    Multiple log files found: {[f.name for f in log_files]}. Selecting latest: '{selected_log_file.name}'."
                )
            else:
                selected_log_file = log_files[0]

            print(
                f"    Processing log file: '{selected_log_file.relative_to(exp_config_path.parent)}'"
            )  # Path relative to seed dir

            metric_value = extract_metric_from_file(
                selected_log_file, args.metric_string
            )
            results_by_exp_name[exp_config_name][seed_val] = metric_value

    # --- Calculate and report mean/std for each experiment configuration name ---
    print(f"\n\n--- Results Summary for Metric: '{args.metric_string}' ---")

    if not results_by_exp_name:
        print("No data was processed or no experiment configurations found.")
        exit(0)

    sorted_exp_names = sorted(results_by_exp_name.keys())

    for exp_config_name in sorted_exp_names:
        seed_data_for_this_exp = results_by_exp_name[exp_config_name]
        print(f"\nExperiment Configuration: {exp_config_name}")

        metric_values_for_this_config = []
        processed_seeds_details = []

        # Iterate through args.seed to check which seeds contributed or were expected for this exp_config_name
        for seed_val in args.seed:
            if (
                seed_val in seed_data_for_this_exp
            ):  # This exp_config was found under this seed
                value = seed_data_for_this_exp[seed_val]
                if value is not None:
                    metric_values_for_this_config.append(value)
                    processed_seeds_details.append(f"Seed {seed_val}: {value:.4f}")
                else:
                    processed_seeds_details.append(
                        f"Seed {seed_val}: No data / Extraction failed"
                    )
            else:
                # This experiment_config_name was not found in this seed's directory
                processed_seeds_details.append(
                    f"Seed {seed_val}: Config '{exp_config_name}' not present"
                )

        if processed_seeds_details:
            print(f"  Data points by seed: [{'; '.join(processed_seeds_details)}]")

        if metric_values_for_this_config:
            num_valid_seeds = len(metric_values_for_this_config)
            mean_metric = np.mean(metric_values_for_this_config)

            if num_valid_seeds > 1:
                std_metric = np.std(
                    metric_values_for_this_config, ddof=0
                )  # ddof=0 for population stdev
            else:
                std_metric = 0.0

            print(
                f"  Mean {args.metric_string} (N={num_valid_seeds}): {mean_metric * 100:.2f} (raw: {mean_metric:.4f})"
            )
            print(
                f"  Std Dev {args.metric_string} (N={num_valid_seeds}): {std_metric * 100:.2f} (raw: {std_metric:.4f})"
            )
        else:
            # This case handles if an exp_config_name was found, but no valid metric values were extracted from any seed for it.
            print(
                f"  No valid '{args.metric_string}' values found across the processed seeds for this configuration."
            )

    print("\nDone.")
