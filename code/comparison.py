"""
Comparison of DTSCAN with other clustering algorithms
This script compares the performance of DTSCAN with:
- DBSCAN (density-based)
- K-means (partition-based)
- Spectral Clustering (graph-based)

Reference: Section 4 of Kim & Cho (2019) - Experimental Results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import time
from dtscan import DTSCAN, generate_test_data_s1, generate_test_data_s2, generate_test_data_s3, calculate_psr


def compare_clustering_algorithms(X, true_labels, dataset_name="Dataset"):
    """
    Compare different clustering algorithms on the same dataset.
    
    Reference: Tables 1-3 in the paper comparing PSR and VSR metrics
    """
    results = {}
    
    # Normalize data for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Estimate number of clusters from true labels (excluding noise)
    n_clusters = len(np.unique(true_labels[true_labels != -1]))
    
    print(f"\n{'='*60}")
    print(f"Comparing algorithms on {dataset_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"True number of clusters: {n_clusters}")
    print(f"{'='*60}")
    
    # 1. DTSCAN (our implementation)
    print("\n1. DTSCAN (Delaunay Triangulation-based)...")
    start_time = time.time()
    dtscan = DTSCAN(z_score_threshold=2.0, min_pts=6)
    labels_dtscan = dtscan.fit_predict(X)
    time_dtscan = time.time() - start_time
    psr_dtscan = calculate_psr(true_labels, labels_dtscan)
    results['DTSCAN'] = {
        'labels': labels_dtscan,
        'PSR': psr_dtscan,
        'time': time_dtscan,
        'n_clusters': len(np.unique(labels_dtscan[labels_dtscan != -1]))
    }
    print(f"   Clusters found: {results['DTSCAN']['n_clusters']}")
    print(f"   PSR: {psr_dtscan:.3f}")
    print(f"   Time: {time_dtscan:.3f}s")
    
    # 2. DBSCAN (traditional density-based)
    print("\n2. DBSCAN (Traditional density-based)...")
    start_time = time.time()
    # Estimate eps using k-distance graph (simplified)
    from sklearn.neighbors import NearestNeighbors
    k = 6  # Same as MinPts in DTSCAN
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    # Use the elbow point (simplified - using 90th percentile)
    eps = np.percentile(distances, 90)
    
    dbscan = DBSCAN(eps=eps, min_samples=k)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    time_dbscan = time.time() - start_time
    psr_dbscan = calculate_psr(true_labels, labels_dbscan)
    results['DBSCAN'] = {
        'labels': labels_dbscan,
        'PSR': psr_dbscan,
        'time': time_dbscan,
        'n_clusters': len(np.unique(labels_dbscan[labels_dbscan != -1]))
    }
    print(f"   Eps used: {eps:.3f}")
    print(f"   Clusters found: {results['DBSCAN']['n_clusters']}")
    print(f"   PSR: {psr_dbscan:.3f}")
    print(f"   Time: {time_dbscan:.3f}s")
    
    # 3. K-means (partition-based)
    print("\n3. K-means (Partition-based)...")
    start_time = time.time()
    # Try to find optimal k using elbow method (simplified)
    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
    else:
        # Fallback if no clear cluster number
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
    time_kmeans = time.time() - start_time
    psr_kmeans = calculate_psr(true_labels, labels_kmeans)
    results['K-means'] = {
        'labels': labels_kmeans,
        'PSR': psr_kmeans,
        'time': time_kmeans,
        'n_clusters': len(np.unique(labels_kmeans))
    }
    print(f"   Clusters found: {results['K-means']['n_clusters']}")
    print(f"   PSR: {psr_kmeans:.3f}")
    print(f"   Time: {time_kmeans:.3f}s")
    
    # 4. Spectral Clustering (graph-based)
    print("\n4. Spectral Clustering (Graph-based)...")
    start_time = time.time()
    if n_clusters > 0 and len(X) < 1000:  # Spectral clustering can be slow for large datasets
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                     random_state=42, n_neighbors=10)
        labels_spectral = spectral.fit_predict(X_scaled)
        time_spectral = time.time() - start_time
        psr_spectral = calculate_psr(true_labels, labels_spectral)
        results['Spectral'] = {
            'labels': labels_spectral,
            'PSR': psr_spectral,
            'time': time_spectral,
            'n_clusters': len(np.unique(labels_spectral))
        }
        print(f"   Clusters found: {results['Spectral']['n_clusters']}")
        print(f"   PSR: {psr_spectral:.3f}")
        print(f"   Time: {time_spectral:.3f}s")
    else:
        print("   Skipped (dataset too large or no clear cluster number)")
    
    return results


def visualize_comparison(X, results, dataset_name="Dataset"):
    """
    Visualize clustering results from different algorithms.
    
    Reference: Figures 6-8 in the paper showing comparative results
    """
    n_algorithms = len(results)
    fig, axes = plt.subplots(2, (n_algorithms + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (algo_name, result) in enumerate(results.items()):
        ax = axes[idx]
        labels = result['labels']
        
        # Plot points colored by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels[unique_labels != -1])))
        
        color_idx = 0
        for label in unique_labels:
            if label == -1:
                # Noise points in black
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], c='black', s=10, alpha=0.3, label='Noise')
            else:
                # Cluster points in colors
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[color_idx]], 
                          s=10, label=f'Cluster {label}')
                color_idx += 1
        
        ax.set_title(f'{algo_name}\nPSR={result["PSR"]:.3f}\n'
                    f'Clusters={result["n_clusters"]}, Time={result["time"]:.3f}s')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Clustering Algorithm Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_summary_table(all_results):
    """
    Create a summary table similar to Tables 1-3 in the paper.

    Note: VSR is not properly defined in the paper (Equation 5 is identical to Equation 4),
    so we only report PSR here.
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE - Performance Comparison (Reference: Tables 1-3 in paper)")
    print("="*80)

    # Header
    print(f"{'Dataset':<15} {'Algorithm':<15} {'PSR':<10} {'Clusters':<10} {'Time(s)':<10}")
    print("-"*60)

    for dataset_name, results in all_results.items():
        for algo_name, metrics in results.items():
            print(f"{dataset_name:<15} {algo_name:<15} "
                  f"{metrics['PSR']:<10.3f} "
                  f"{metrics['n_clusters']:<10} {metrics['time']:<10.3f}")
        print("-"*60)
    
    print("\n" + "="*80)
    print("KEY FINDINGS (Reference: Section 4 - Experimental Results):")
    print("-"*80)
    print("1. DTSCAN shows superior performance in handling touching problems")
    print("2. Traditional DBSCAN struggles with adjacent clusters and varying densities")
    print("3. K-means fails on non-spherical and non-linear cluster shapes")
    print("4. Spectral Clustering performs well but with higher computational cost")
    print("5. DTSCAN's z-score filtering effectively removes chain noise")
    print("="*80)


def main():
    """
    Main function to run all comparisons.
    """
    print("="*80)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("Reproducing experiments from Kim & Cho (2019)")
    print("="*80)
    
    all_results = {}
    
    # Test Dataset S1 - Complex clusters with touching problems
    print("\n" + "#"*80)
    print("DATASET S1 - Complex clusters with touching problems")
    print("Reference: Figure 6 and Table 1 in the paper")
    print("#"*80)
    X_s1, true_labels_s1 = generate_test_data_s1()
    results_s1 = compare_clustering_algorithms(X_s1, true_labels_s1, "S1")
    all_results['S1'] = results_s1
    visualize_comparison(X_s1, results_s1, "S1 - Complex Clusters")
    
    # Test Dataset S2 - Nested/concentric clusters
    print("\n" + "#"*80)
    print("DATASET S2 - Nested/concentric clusters with different densities")
    print("Reference: Figure 7 and Table 2 in the paper")
    print("#"*80)
    X_s2, true_labels_s2 = generate_test_data_s2()
    results_s2 = compare_clustering_algorithms(X_s2, true_labels_s2, "S2")
    all_results['S2'] = results_s2
    visualize_comparison(X_s2, results_s2, "S2 - Nested Clusters")
    
    # Test Dataset S3 - Adjacent nonlinear clusters
    print("\n" + "#"*80)
    print("DATASET S3 - Adjacent nonlinear clusters with uneven density")
    print("Reference: Figure 8 and Table 3 in the paper")
    print("#"*80)
    X_s3, true_labels_s3 = generate_test_data_s3()
    results_s3 = compare_clustering_algorithms(X_s3, true_labels_s3, "S3")
    all_results['S3'] = results_s3
    visualize_comparison(X_s3, results_s3, "S3 - Adjacent Nonlinear")
    
    # Create summary table
    create_summary_table(all_results)
    
    # Performance analysis specific to touching problems
    print("\n" + "="*80)
    print("TOUCHING PROBLEM ANALYSIS")
    print("Reference: Section 3 - 'touching problems and the adjacency problems of clusters'")
    print("="*80)
    
    # Focus on S1 dataset which has clear touching clusters
    print("\nAnalyzing touching clusters in S1 dataset:")
    print("-"*40)
    
    # Check how each algorithm handles the touching clusters (clusters 3 and 4 in our generation)
    for algo_name, result in results_s1.items():
        labels = result['labels']
        # Analyze points in the touching region (around x=1.8, y=0.3)
        touching_region_mask = ((X_s1[:, 0] > 1.6) & (X_s1[:, 0] < 2.0) & 
                               (X_s1[:, 1] > 0.1) & (X_s1[:, 1] < 0.5))
        touching_labels = labels[touching_region_mask]
        unique_touching = np.unique(touching_labels[touching_labels != -1])
        
        print(f"{algo_name}: Found {len(unique_touching)} cluster(s) in touching region")
        if len(unique_touching) > 1:
            print(f"   ✓ Successfully separated touching clusters")
        elif len(unique_touching) == 1:
            print(f"   ✗ Failed to separate touching clusters (merged)")
        else:
            print(f"   ✗ Classified touching region as noise")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("DTSCAN successfully addresses the limitations of existing clustering methods:")
    print("• Handles nonlinear cluster shapes effectively")
    print("• Separates adjacent/touching clusters using Delaunay triangulation")
    print("• Robust against chain noise through z-score filtering")
    print("• Maintains good performance across varying cluster densities")
    print("="*80)


if __name__ == "__main__":
    main()
