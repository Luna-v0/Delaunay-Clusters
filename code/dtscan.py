"""
DTSCAN: Delaunay Triangulation-Based Spatial Clustering
Corrected implementation based on the paper:
"Delaunay Triangulation-Based Spatial Clustering Technique for Enhanced Adjacent Boundary 
Detection and Segmentation of LiDAR 3D Point Clouds"
by Jongwon Kim and Jeongho Cho (2019)

Key corrections:
1. Proper z-score filtering (remove outliers with HIGH z-scores)
2. Correct graph construction from Delaunay triangulation
3. DBSCAN-like clustering on the filtered graph
"""

import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict


class DTSCAN:
    """
    Delaunay Triangulation-based Spatial Clustering of Applications with Noise (DTSCAN)

    This clustering algorithm combines Delaunay triangulation with DBSCAN's density-based
    clustering mechanism to handle:
    - Nonlinear shapes
    - Irregular density
    - Touching problems of adjacent clusters
    - Various types of noise (background and chain noise)

    Reference: Section 3 of Kim & Cho (2019)
    """

    def __init__(self,
                 z_score_threshold: float = 2.0,
                 min_pts: int = 6,
                 area_threshold: Optional[float] = None,
                 length_threshold: Optional[float] = None):
        """
        Initialize DTSCAN clustering algorithm.

        Parameters:
        -----------
        z_score_threshold : float
            Threshold for z-score normalization (default: 2.0)
            Used for removing edges/triangles that are outliers
            Reference: Equations (2) and (3) in the paper

        min_pts : int
            Minimum number of neighboring nodes required to form a cluster
            Similar to DBSCAN's MinPts parameter
            Reference: Section 3, clustering process step 2

        area_threshold : float, optional
            Custom threshold for triangle areas (overrides z_score_threshold)

        length_threshold : float, optional
            Custom threshold for edge lengths (overrides z_score_threshold)
        """
        self.z_score_threshold = z_score_threshold
        self.min_pts = min_pts
        self.area_threshold = area_threshold
        self.length_threshold = length_threshold

        # Storage for intermediate results
        self.triangulation = None
        self.graph_edges = None
        self.filtered_graph = None
        self.labels = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DTSCAN clustering on the input data.

        Parameters:
        -----------
        X : np.ndarray
            Input data points of shape (n_samples, n_features)
            For 2D data: shape (n, 2)
            For 3D data: shape (n, 3)

        Returns:
        --------
        labels : np.ndarray
            Cluster labels for each point (-1 indicates noise)

        Reference: Algorithm flow in Section 3
        """
        # Step 1: Perform Delaunay triangulation
        # Reference: Section 2 - "Delaunay triangulation splits the space by connecting
        # the points on the plane with triangles"
        print("Step 1: Performing Delaunay triangulation...")
        self.triangulation = Delaunay(X)

        # Step 2: Build graph from triangulation
        # Reference: Section 3 - "DTSCAN extends all of the edges and vertices based on
        # the triangulation to a graph-based clustering technique"
        print("Step 2: Building graph from triangulation...")
        self.graph_edges = self._build_graph_from_triangulation(X)

        # Step 3: Remove global effects using z-score normalization
        # Reference: Equations (2) and (3) in the paper
        print("Step 3: Removing global effects (filtering outlier edges/triangles)...")
        self.filtered_graph = self._remove_global_effects(X)

        # Step 4: Apply density-based clustering mechanism
        # Reference: Section 3 - "The proposed DTSCAN utilizes the existing DBSCAN
        # clustering mechanism to form clusters"
        print("Step 4: Applying density-based clustering...")
        self.labels = self._density_based_clustering(len(X))

        return self.labels

    def _build_graph_from_triangulation(self, X: np.ndarray) -> Dict[int, Set[int]]:
        """
        Build an undirected graph from Delaunay triangulation.

        Each point becomes a vertex, and edges connect points that share
        a triangle in the Delaunay triangulation.

        Reference: Section 3 - Graph construction from triangulation
        """
        graph = defaultdict(set)

        # For each simplex (triangle in 2D, tetrahedron in 3D)
        for simplex in self.triangulation.simplices:
            # Connect all pairs of vertices in the simplex
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    v1, v2 = simplex[i], simplex[j]
                    graph[v1].add(v2)
                    graph[v2].add(v1)

        return dict(graph)

    def _calculate_triangle_areas(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the area of each triangle in the triangulation.

        For 2D triangles: Use cross product formula
        For 3D triangles: Use vector cross product norm

        Reference: Used in Equation (2) for area-based filtering
        """
        areas = []

        for simplex in self.triangulation.simplices:
            if X.shape[1] == 2:
                # 2D case: Calculate triangle area using cross product
                # Area = 0.5 * |det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])|
                points = X[simplex]
                # Shoelace formula
                area = 0.5 * abs(
                    (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]) -
                    (points[2, 0] - points[0, 0]) *
                    (points[1, 1] - points[0, 1])
                )
            else:
                # 3D case: Calculate triangle area using vector cross product
                points = X[simplex[:3]]  # Take first 3 points for tetrahedron
                v1 = points[1] - points[0]
                v2 = points[2] - points[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))

            areas.append(area)

        return np.array(areas)

    def _calculate_edge_lengths(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the length of each unique edge in the triangulation.

        Reference: Used in Equation (3) for edge-based filtering
        """
        edge_lengths = []
        edges_seen = set()

        for simplex in self.triangulation.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    if edge not in edges_seen:
                        edges_seen.add(edge)
                        length = np.linalg.norm(X[edge[0]] - X[edge[1]])
                        edge_lengths.append(length)

        return np.array(edge_lengths)

    def _remove_global_effects(self, X: np.ndarray) -> Dict[int, Set[int]]:
        """
        Remove edges and triangles that are outliers based on z-score normalization.

        This step filters out:
        - Triangles with unusually large areas (likely between clusters)
        - Edges with unusually long lengths (likely noise connections)

        Reference: Section 3, Equations (2) and (3)
        """
        # Calculate areas and apply z-score normalization
        # Note: The paper mentions removing triangles with "relatively wide or long nodes"
        # So we filter out triangles with LARGE areas (positive z-scores)
        areas = self._calculate_triangle_areas(X)

        # Calculate mean and std for areas
        area_mean = np.mean(areas)
        area_std = np.std(areas)

        # Calculate z-scores according to Equation (2) in the paper
        # Formula: (mean - value) / std
        # Large/wide triangles get negative z-scores and should be removed
        area_z_scores = (area_mean - areas) / (area_std + 1e-10)

        # Build edge length mapping
        edge_to_length = {}
        for simplex in self.triangulation.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    if edge not in edge_to_length:
                        length = np.linalg.norm(X[edge[0]] - X[edge[1]])
                        edge_to_length[edge] = length

        # Calculate edge length statistics
        all_lengths = list(edge_to_length.values())
        length_mean = np.mean(all_lengths)
        length_std = np.std(all_lengths)

        # Determine thresholds (typically z_score_threshold = 2.0)
        area_thresh = self.area_threshold if self.area_threshold else self.z_score_threshold
        length_thresh = self.length_threshold if self.length_threshold else self.z_score_threshold

        # Filter triangles based on area z-scores (remove large triangles)
        # With corrected z-score: large triangles have negative z-scores
        # Keep triangles where z-score >= -threshold (i.e., not too large)
        valid_simplices = []
        for idx, (simplex, z_score) in enumerate(zip(self.triangulation.simplices, area_z_scores)):
            # Keep triangles with small to medium areas (z-score >= -threshold)
            if z_score >= -area_thresh:
                valid_simplices.append(simplex)

        # Rebuild graph with filtered triangles and edges
        filtered_graph = defaultdict(set)

        for simplex in valid_simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    length = edge_to_length[edge]

                    # Calculate z-score for this edge length according to Equation (3)
                    # Formula: (mean - value) / std
                    # Long edges get negative z-scores and should be removed
                    edge_z_score = (length_mean - length) / \
                        (length_std + 1e-10)

                    # Keep edges with reasonable lengths (not too long)
                    # Keep edges where z-score >= -threshold (i.e., not too long)
                    if edge_z_score >= -length_thresh:
                        v1, v2 = simplex[i], simplex[j]
                        filtered_graph[v1].add(v2)
                        filtered_graph[v2].add(v1)

        # Ensure all points are in the graph (even if isolated)
        for i in range(len(X)):
            if i not in filtered_graph:
                filtered_graph[i] = set()

        # Convert defaultdict to regular dict
        return dict(filtered_graph)

    def _density_based_clustering(self, n_points: int) -> np.ndarray:
        """
        Apply DBSCAN-like density-based clustering on the filtered graph.

        Key differences from standard DBSCAN:
        - Uses graph connectivity instead of epsilon radius
        - Neighbors are directly connected nodes in the graph

        Reference: Section 3 - Clustering process steps 1-4
        """
        labels = np.full(n_points, -1)  # Initialize all points as noise (-1)
        cluster_id = 0
        visited = set()

        for point_id in range(n_points):
            if point_id in visited:
                continue

            visited.add(point_id)

            # Step 1: Get neighboring nodes (directly connected in graph)
            # Reference: "A neighboring node of point pi in a triangle is a set of
            # edge graphs directly connected to pi"
            neighbors = self.filtered_graph.get(point_id, set())

            # Step 2: Check if point has enough neighbors to form a cluster
            # Reference: "If the number of elements in the set of connected neighboring
            # nodes is greater than or equal to MinPts"
            if len(neighbors) < self.min_pts:
                continue  # Point remains as noise

            # Start a new cluster
            labels[point_id] = cluster_id

            # Step 3: Expand cluster using BFS
            # Reference: "(1)-(2) is repeated for neighboring nodes of point pi and
            # the cluster is expanded by searching for neighboring nodes"
            queue = deque(neighbors)

            while queue:
                neighbor = queue.popleft()

                if neighbor in visited:
                    if labels[neighbor] == -1:  # Was noise, now border point
                        labels[neighbor] = cluster_id
                    continue

                visited.add(neighbor)
                labels[neighbor] = cluster_id

                # Get neighbors of the neighbor
                neighbor_neighbors = self.filtered_graph.get(neighbor, set())

                # If neighbor is a core point, add its neighbors to queue
                if len(neighbor_neighbors) >= self.min_pts:
                    queue.extend(neighbor_neighbors)

            cluster_id += 1

        return labels

    def visualize_process(self, X: np.ndarray, figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize the DTSCAN clustering process in stages.

        Shows:
        1. Original Delaunay triangulation
        2. Filtered graph after removing global effects
        3. Final clustering results

        Only works for 2D data.
        """
        if X.shape[1] != 2:
            print("Visualization only supported for 2D data")
            return

        if self.labels is None:
            print("Please run fit_predict() first")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Original Delaunay triangulation
        axes[0].triplot(
            X[:, 0], X[:, 1], self.triangulation.simplices, 'b-', alpha=0.3, linewidth=0.5)
        axes[0].plot(X[:, 0], X[:, 1], 'ko', markersize=3)
        axes[0].set_title('Original Delaunay Triangulation')
        axes[0].set_aspect('equal')

        # Plot 2: Filtered graph
        axes[1].plot(X[:, 0], X[:, 1], 'ko', markersize=3)
        for node, neighbors in self.filtered_graph.items():
            for neighbor in neighbors:
                if node < neighbor:  # Draw each edge only once
                    axes[1].plot([X[node, 0], X[neighbor, 0]],
                                 [X[node, 1], X[neighbor, 1]],
                                 'b-', alpha=0.5, linewidth=0.5)
        axes[1].set_title('Graph After Removing Global Effects')
        axes[1].set_aspect('equal')

        # Plot 3: Final clustering
        unique_labels = np.unique(self.labels)
        colors = plt.cm.rainbow(np.linspace(
            0, 1, len(unique_labels[unique_labels != -1])))

        color_idx = 0
        for label in unique_labels:
            if label == -1:
                # Noise points in black
                mask = self.labels == label
                axes[2].plot(X[mask, 0], X[mask, 1], 'k.',
                             markersize=3, alpha=0.3)
            else:
                # Cluster points in colors
                mask = self.labels == label
                axes[2].plot(X[mask, 0], X[mask, 1], '.',
                             color=colors[color_idx], markersize=5)
                color_idx += 1

        axes[2].set_title(
            f'DTSCAN Clustering Result ({len(unique_labels[unique_labels != -1])} clusters)')
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()


def calculate_psr(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate PSR (Point Score Range) metric.

    Reference: Equation (4) in the paper

    Note: The paper's Equation (5) claiming to define VSR is identical to Equation (4),
    so VSR is not properly defined. We only compute PSR here.

    PSR measures clustering quality using IoU (Jaccard Index) between true and predicted clusters.
    """
    # Remove noise labels for evaluation
    mask = true_labels != -1
    true_labels_clean = true_labels[mask]
    pred_labels_clean = pred_labels[mask]

    unique_true = np.unique(true_labels_clean)
    unique_pred = np.unique(pred_labels_clean[pred_labels_clean != -1])

    if len(unique_pred) == 0:
        return 0.0

    psr_scores = []

    for true_cluster in unique_true:
        true_mask = true_labels_clean == true_cluster
        best_psr = 0

        for pred_cluster in unique_pred:
            pred_mask = pred_labels_clean == pred_cluster

            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)

            if union > 0:
                # Using IoU (Jaccard Index) as PSR metric
                psr = intersection / union
                best_psr = max(best_psr, psr)

        psr_scores.append(best_psr)

    avg_psr = np.mean(psr_scores) if psr_scores else 0
    return avg_psr


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("DTSCAN Implementation - Testing with Synthetic Data")
    print("Based on Kim & Cho (2019)")
    print("="*80)

    # Test with S1-like data
    print("\n" + "="*40)
    print("Testing with S1-like data (complex clusters with touching problems)")
    print("="*40)

    X_s1, true_labels_s1 = generate_test_data_s1()

    # Initialize DTSCAN with parameters from the paper
    # Paper uses MinPts=6 for the examples
    dtscan = DTSCAN(z_score_threshold=2.0, min_pts=6)

    # Perform clustering
    labels_s1 = dtscan.fit_predict(X_s1)

    # Calculate metrics
    psr = calculate_psr(true_labels_s1, labels_s1)

    print(
        f"Number of clusters found: {len(np.unique(labels_s1[labels_s1 != -1]))}")
    print(f"Number of noise points: {np.sum(labels_s1 == -1)}")
    print(f"PSR (Point Score Range): {psr:.3f}")

    # Visualize the process
    print("\nVisualizing clustering process...")
    dtscan.visualize_process(X_s1)

    # Test with S2-like data
    print("\n" + "="*40)
    print("Testing with S2-like data (nested clusters)")
    print("="*40)

    X_s2, true_labels_s2 = generate_test_data_s2()
    dtscan_s2 = DTSCAN(z_score_threshold=2.0, min_pts=6)
    labels_s2 = dtscan_s2.fit_predict(X_s2)

    psr = calculate_psr(true_labels_s2, labels_s2)
    print(
        f"Number of clusters found: {len(np.unique(labels_s2[labels_s2 != -1]))}")
    print(f"Number of noise points: {np.sum(labels_s2 == -1)}")
    print(f"PSR: {psr:.3f}")

    # Test with S3-like data
    print("\n" + "="*40)
    print("Testing with S3-like data (adjacent nonlinear clusters)")
    print("="*40)

    X_s3, true_labels_s3 = generate_test_data_s3()
    dtscan_s3 = DTSCAN(z_score_threshold=2.0, min_pts=6)
    labels_s3 = dtscan_s3.fit_predict(X_s3)

    psr = calculate_psr(true_labels_s3, labels_s3)
    print(
        f"Number of clusters found: {len(np.unique(labels_s3[labels_s3 != -1]))}")
    print(f"Number of noise points: {np.sum(labels_s3 == -1)}")
    print(f"PSR: {psr:.3f}")

    print("\n" + "="*80)
    print("Implementation complete!")
    print("="*80)
