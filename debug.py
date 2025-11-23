
import sys
import os
# append code directory to path - must be before importing dtscan
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'code'))

from dtscan import DTSCAN, psr
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

def load_dataset(file_path):
    """Loads a dataset from a text file."""
    data = np.loadtxt(file_path)
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    return X, true_labels


def find_best_parameters(dataset_name, file_path, expected_clusters):
    """
    Runs a grid search for DTSCAN parameters for a single dataset and prints the best results.
    This version focuses on finding the params that give the *correct number of clusters* first,
    then maximizing PSR.
    """
    print(f"--- Searching Best Parameters for {dataset_name} ---")
    X, true_labels = load_dataset(file_path)

    # Define the grid of parameters to search
    min_pts_values = [4, 5, 6, 7, 8]
    # More granular search in the promising range
    threshold_values = np.linspace(-4.0, 4.0, 20)

    best_score_for_correct_clusters = -1
    best_params_for_correct_clusters = {}

    for min_pts in min_pts_values:
        # Decouple area and length thresholds for a more thorough search
        for area_th in threshold_values:
            for length_th in threshold_values:

                # Initialize and run DTSCAN
                dtscan = DTSCAN(
                    MinPts=min_pts, area_threshold=area_th, length_threshold=length_th)

                # Suppress prints during grid search
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                pred_labels = dtscan.fit_predict(X)
                sys.stdout = old_stdout

                n_clusters_found = len(
                    np.unique(pred_labels[pred_labels != -1]))

                # Prioritize solutions with the correct number of clusters
                if n_clusters_found == expected_clusters:
                    score = psr(true_labels, pred_labels)
                    if score > best_score_for_correct_clusters:
                        best_score_for_correct_clusters = score
                        best_params_for_correct_clusters = {
                            "min_pts": min_pts,
                            "area_threshold": area_th,
                            "length_threshold": length_th,
                            "n_clusters_found": n_clusters_found
                        }

    if best_params_for_correct_clusters:
        print(
            f"Best Parameters Found for {dataset_name} (matching cluster count):")
        params = best_params_for_correct_clusters
        print(f"  MinPts: {params.get('min_pts', 'N/A')}")
        print(f"  Area Threshold: {params.get('area_threshold', 'N/A'):}")
        print(
            f"  Length Threshold: {params.get('length_threshold', 'N/A'):}")
        print(f"  Clusters found: {params.get('n_clusters_found', 'N/A')}")
        print(f"  Best PSR Score: {best_score_for_correct_clusters:}")
    else:
        print(
            f"Could not find any parameter combination that resulted in {expected_clusters} clusters.")

    print("-" * (len(dataset_name) + 30))


if __name__ == "__main__":
    # Expected results from Instructions.md
    expected_results = {
        "S1": {"clusters": 7, "psr": 0.999},
        "S2": {"clusters": 3, "psr": 0.934},  # Note: paper finds a bit more
        "S3": {"clusters": 2, "psr": 1.000},
    }

    # Focusing on S3 as requested

    for name, expected_result in expected_results.items():
        path = f"datasets/{name}.txt"
        print(f"\nPerforming focused search for {name}...")
        print(
            f"Expected for {name}: {expected_result['clusters']} clusters, PSR {expected_result['psr']:.3f}")
        find_best_parameters(name, path, expected_result['clusters'])
