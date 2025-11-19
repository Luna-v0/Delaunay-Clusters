"""
Hyperparameter Optimization for DTSCAN
This script performs grid search to find optimal parameters for the DTSCAN algorithm.

We'll search over:
- z_score_threshold: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
- min_pts: [3, 4, 5, 6, 7, 8, 10, 12]

Evaluation metrics:
- PSR (Point Score Range): Higher is better (IoU-based clustering quality)
- Number of clusters: Should match expected count
- Combined score: Weighted combination of PSR and cluster count accuracy

Note: VSR is not used as it's not properly defined in the paper.
"""

import numpy as np
import pandas as pd
from itertools import product
import time
from dtscan import (
    DTSCAN,
    generate_test_data_s1,
    generate_test_data_s2,
    generate_test_data_s3,
    calculate_psr
)
from test_3d import generate_3d_pedestrians


def grid_search_dtscan(X, true_labels, expected_n_clusters,
                       z_thresholds=None, min_pts_values=None,
                       dataset_name="Dataset"):
    """
    Perform grid search over DTSCAN hyperparameters.

    Parameters:
    -----------
    X : np.ndarray
        Input data points
    true_labels : np.ndarray
        Ground truth labels
    expected_n_clusters : int
        Expected number of clusters
    z_thresholds : list
        Z-score threshold values to try
    min_pts_values : list
        MinPts values to try
    dataset_name : str
        Name of dataset for reporting

    Returns:
    --------
    results_df : pd.DataFrame
        Results for all parameter combinations
    best_params : dict
        Best parameters found
    """

    if z_thresholds is None:
        z_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    if min_pts_values is None:
        min_pts_values = [3, 4, 5, 6, 7, 8, 10, 12]

    print(f"\n{'='*80}")
    print(f"Grid Search for {dataset_name}")
    print(f"{'='*80}")
    print(f"Data shape: {X.shape}")
    print(f"Expected clusters: {expected_n_clusters}")
    print(f"Z-score thresholds to try: {z_thresholds}")
    print(f"MinPts values to try: {min_pts_values}")
    print(f"Total combinations: {len(z_thresholds) * len(min_pts_values)}")
    print(f"{'='*80}\n")

    results = []
    total_combinations = len(z_thresholds) * len(min_pts_values)
    current = 0

    for z_thresh, min_pts in product(z_thresholds, min_pts_values):
        current += 1

        try:
            # Run DTSCAN with these parameters
            start_time = time.time()
            dtscan = DTSCAN(z_score_threshold=z_thresh, min_pts=min_pts)
            labels = dtscan.fit_predict(X)
            elapsed_time = time.time() - start_time

            # Calculate metrics
            n_clusters = len(np.unique(labels[labels != -1]))
            n_noise = np.sum(labels == -1)
            psr = calculate_psr(true_labels, labels)

            # Cluster count error
            cluster_error = abs(n_clusters - expected_n_clusters)

            # Combined score: weighted combination
            # Perfect score: PSR=1.0, cluster_error=0
            # We want to maximize PSR while minimizing cluster_error
            cluster_penalty = cluster_error / max(expected_n_clusters, 1)
            combined_score = (0.7 * psr + 0.3 * (1 - min(cluster_penalty, 1)))

            results.append({
                'z_threshold': z_thresh,
                'min_pts': min_pts,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'PSR': psr,
                'cluster_error': cluster_error,
                'combined_score': combined_score,
                'time': elapsed_time,
                'status': 'success'
            })

            # Progress update
            if current % 10 == 0 or current == total_combinations:
                print(f"Progress: {current}/{total_combinations} - "
                      f"Last: z={z_thresh}, min_pts={min_pts}, "
                      f"clusters={n_clusters}, PSR={psr:.3f}")

        except Exception as e:
            results.append({
                'z_threshold': z_thresh,
                'min_pts': min_pts,
                'n_clusters': 0,
                'n_noise': 0,
                'PSR': 0.0,
                'cluster_error': expected_n_clusters,
                'combined_score': 0.0,
                'time': 0.0,
                'status': f'error: {str(e)}'
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best parameters based on different criteria
    best_combined = results_df.loc[results_df['combined_score'].idxmax()]
    best_psr = results_df.loc[results_df['PSR'].idxmax()]
    best_cluster_match = results_df.loc[results_df['cluster_error'].idxmin()]

    print(f"\n{'='*80}")
    print(f"GRID SEARCH RESULTS - {dataset_name}")
    print(f"{'='*80}")

    print(f"\nBest by Combined Score:")
    print(f"  z_threshold={best_combined['z_threshold']}, min_pts={best_combined['min_pts']}")
    print(f"  Clusters: {int(best_combined['n_clusters'])} (expected: {expected_n_clusters})")
    print(f"  PSR: {best_combined['PSR']:.3f}")
    print(f"  Combined Score: {best_combined['combined_score']:.3f}")

    print(f"\nBest by PSR:")
    print(f"  z_threshold={best_psr['z_threshold']}, min_pts={best_psr['min_pts']}")
    print(f"  Clusters: {int(best_psr['n_clusters'])} (expected: {expected_n_clusters})")
    print(f"  PSR: {best_psr['PSR']:.3f}")

    print(f"\nBest by Cluster Count Match:")
    print(f"  z_threshold={best_cluster_match['z_threshold']}, min_pts={best_cluster_match['min_pts']}")
    print(f"  Clusters: {int(best_cluster_match['n_clusters'])} (expected: {expected_n_clusters})")
    print(f"  PSR: {best_cluster_match['PSR']:.3f}")

    # Statistics
    print(f"\n{'='*80}")
    print(f"STATISTICS")
    print(f"{'='*80}")
    print(f"Successful runs: {len(results_df[results_df['status'] == 'success'])}/{len(results_df)}")
    print(f"PSR range: [{results_df['PSR'].min():.3f}, {results_df['PSR'].max():.3f}]")
    print(f"Cluster count range: [{int(results_df['n_clusters'].min())}, {int(results_df['n_clusters'].max())}]")
    print(f"Combined score range: [{results_df['combined_score'].min():.3f}, {results_df['combined_score'].max():.3f}]")

    best_params = {
        'best_combined': {
            'z_threshold': best_combined['z_threshold'],
            'min_pts': int(best_combined['min_pts']),
            'score': best_combined['combined_score']
        },
        'best_psr': {
            'z_threshold': best_psr['z_threshold'],
            'min_pts': int(best_psr['min_pts']),
            'psr': best_psr['PSR']
        },
        'best_cluster_match': {
            'z_threshold': best_cluster_match['z_threshold'],
            'min_pts': int(best_cluster_match['min_pts']),
            'error': best_cluster_match['cluster_error']
        }
    }

    return results_df, best_params


def analyze_parameter_sensitivity(results_df, dataset_name="Dataset"):
    """
    Analyze how each parameter affects performance.
    """
    print(f"\n{'='*80}")
    print(f"PARAMETER SENSITIVITY ANALYSIS - {dataset_name}")
    print(f"{'='*80}")

    # Effect of z_threshold (averaging over min_pts)
    print("\nEffect of z_threshold (averaged over all min_pts values):")
    z_analysis = results_df.groupby('z_threshold').agg({
        'PSR': 'mean',
        'n_clusters': 'mean',
        'combined_score': 'mean'
    }).round(3)
    print(z_analysis)

    # Effect of min_pts (averaging over z_threshold)
    print("\nEffect of min_pts (averaged over all z_threshold values):")
    min_pts_analysis = results_df.groupby('min_pts').agg({
        'PSR': 'mean',
        'n_clusters': 'mean',
        'combined_score': 'mean'
    }).round(3)
    print(min_pts_analysis)

    # Top 10 parameter combinations
    print("\nTop 10 Parameter Combinations (by combined score):")
    top_10 = results_df.nlargest(10, 'combined_score')[
        ['z_threshold', 'min_pts', 'n_clusters', 'PSR', 'combined_score']
    ].round(3)
    print(top_10.to_string(index=False))


def run_comprehensive_grid_search():
    """
    Run grid search on all datasets.
    """
    print("="*80)
    print("COMPREHENSIVE HYPERPARAMETER OPTIMIZATION FOR DTSCAN")
    print("="*80)

    all_results = {}
    all_best_params = {}

    # Dataset S1
    print("\n" + "#"*80)
    print("DATASET S1 - Complex Clusters with Touching Problems")
    print("#"*80)
    X_s1, true_labels_s1 = generate_test_data_s1()
    expected_clusters_s1 = len(np.unique(true_labels_s1[true_labels_s1 != -1]))
    results_s1, best_s1 = grid_search_dtscan(
        X_s1, true_labels_s1, expected_clusters_s1, dataset_name="S1"
    )
    analyze_parameter_sensitivity(results_s1, "S1")
    all_results['S1'] = results_s1
    all_best_params['S1'] = best_s1

    # Dataset S2
    print("\n" + "#"*80)
    print("DATASET S2 - Nested/Concentric Clusters")
    print("#"*80)
    X_s2, true_labels_s2 = generate_test_data_s2()
    expected_clusters_s2 = len(np.unique(true_labels_s2[true_labels_s2 != -1]))
    results_s2, best_s2 = grid_search_dtscan(
        X_s2, true_labels_s2, expected_clusters_s2, dataset_name="S2"
    )
    analyze_parameter_sensitivity(results_s2, "S2")
    all_results['S2'] = results_s2
    all_best_params['S2'] = best_s2

    # Dataset S3
    print("\n" + "#"*80)
    print("DATASET S3 - Adjacent Nonlinear Clusters")
    print("#"*80)
    X_s3, true_labels_s3 = generate_test_data_s3()
    expected_clusters_s3 = len(np.unique(true_labels_s3[true_labels_s3 != -1]))
    results_s3, best_s3 = grid_search_dtscan(
        X_s3, true_labels_s3, expected_clusters_s3, dataset_name="S3"
    )
    analyze_parameter_sensitivity(results_s3, "S3")
    all_results['S3'] = results_s3
    all_best_params['S3'] = best_s3

    # 3D Datasets
    print("\n" + "#"*80)
    print("3D POINT CLOUDS - Pedestrian Separation")
    print("#"*80)

    for level in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"3D Level {level} (λ={[4, 2, 1][level-1]})")
        print(f"{'='*60}")
        X_3d, true_labels_3d = generate_3d_pedestrians(separation_level=level)
        expected_clusters_3d = len(np.unique(true_labels_3d[true_labels_3d != -1]))

        # For 3D, try higher min_pts values
        results_3d, best_3d = grid_search_dtscan(
            X_3d, true_labels_3d, expected_clusters_3d,
            z_thresholds=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            min_pts_values=[5, 7, 10, 12, 15, 20],
            dataset_name=f"3D-Level{level}"
        )
        analyze_parameter_sensitivity(results_3d, f"3D-Level{level}")
        all_results[f'3D-L{level}'] = results_3d
        all_best_params[f'3D-L{level}'] = best_3d

    # Summary of all datasets
    print("\n" + "="*80)
    print("SUMMARY: BEST PARAMETERS FOR ALL DATASETS")
    print("="*80)

    for dataset_name, best_params in all_best_params.items():
        print(f"\n{dataset_name}:")
        print(f"  Best Combined: z_threshold={best_params['best_combined']['z_threshold']}, "
              f"min_pts={best_params['best_combined']['min_pts']}, "
              f"score={best_params['best_combined']['score']:.3f}")

    # Save results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    for dataset_name, results_df in all_results.items():
        filename = f"grid_search_results_{dataset_name}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Saved {filename}")

    return all_results, all_best_params


if __name__ == "__main__":
    results, best_params = run_comprehensive_grid_search()

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80)
    print("\nCheck the CSV files for detailed results.")
    print("\nKey Insight: Compare the best parameters found with the paper's")
    print("suggested values (z_threshold=2.0, min_pts=6)")
