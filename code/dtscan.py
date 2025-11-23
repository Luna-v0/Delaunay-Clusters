"""
DTSCAN: Delaunay Triangulation-Based Spatial Clustering of Applications with Noise

Implementation based on the paper:
"Delaunay Triangulation-Based Spatial Clustering Technique for Enhanced Adjacent Boundary
Detection and Segmentation of LiDAR 3D Point Clouds"
by Jongwon Kim and Jeongho Cho, Sensors 2019, 19, 3926
DOI: 10.3390/s19183926

ALGORITHM OVERVIEW:
==================
DTSCAN combines Delaunay triangulation with DBSCAN's density-based clustering to address
challenges in complex spatial data:
- Nonlinear cluster shapes
- Irregular cluster densities
- Touching/adjacent cluster problems
- Background noise and chain noise

MAIN STEPS (Section 3 of the paper):
=====================================
1. Delaunay Triangulation: Partition space into triangles connecting data points
   - Creates graph structure with vertices (points) and edges (triangle sides)

2. Remove Global Effects: Filter out outlier triangles and edges using z-score normalization
   - Equation (2): AreaZ = (Area_mean - Area) / sqrt(Area_var)
   - Equation (3): LengthZ = (Length_mean - Length) / sqrt(Length_var)
   - Remove triangles/edges with negative z-scores (large/long outliers)

3. Density-Based Clustering: Apply DBSCAN mechanism on filtered graph
   - For each point, count neighbors (connected nodes in graph)
   - Points with >= MinPts neighbors form clusters
   - Expand clusters by breadth-first search

4. Evaluation: Use PSR (Point Score Range) metric based on IoU/Jaccard Index
   - Equation (4): PSR measures cluster similarity

KEY TERMINOLOGY:
================
- Q: Set of input points {p1, ..., pN}
- DT(Q): Delaunay triangulation of Q, resulting in triangles {T1, ..., TH}
- MinPts: Minimum number of neighbors for a core point (ONLY hyperparameter)
- AreaZ, LengthZ: Z-score normalized area and edge length
"""

import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass


@dataclass
class Triangle:
    """
    Represents a triangle in the Delaunay triangulation.

    Reference: Used in Section 3 for computing triangle areas and edge lengths
    """
    points: np.ndarray  # Shape (3, n_features) - the actual coordinates of the triangle vertices
    vertex_indices: np.ndarray  # Shape (3,) - indices of vertices in original point set

    def compute_area(self) -> float:
        """
        Calculate the area of the triangle.

        For 2D: Uses the cross product formula (Shoelace formula)
        For 3D: Uses vector cross product norm

        Reference: Equation (2) - Area is used for z-score normalization
        """
        if self.points.shape[1] == 2:
            # 2D case: Shoelace formula
            p = self.points
            area = 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) -
                (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )
        else:
            # 3D case: Use cross product of two edge vectors
            v1 = self.points[1] - self.points[0]
            v2 = self.points[2] - self.points[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        return area

    def compute_edge_lengths(self) -> np.ndarray:
        """
        Calculate the lengths of all three edges of the triangle.

        Returns:
        --------
        edge_lengths : np.ndarray
            Array of shape (3,) containing the three edge lengths

        Reference: Equation (3) - Edge lengths are used for z-score normalization
        """
        lengths = np.array([
            np.linalg.norm(self.points[1] - self.points[0]),  # Edge 0-1
            np.linalg.norm(self.points[2] - self.points[1]),  # Edge 1-2
            np.linalg.norm(self.points[0] - self.points[2])   # Edge 2-0
        ])
        return lengths


def z_area(areas: np.ndarray) -> np.ndarray:
    """
    Calculate z-score normalization for triangle areas.

    According to Equation (2) in the paper:
    AreaZ = (Area_mean(DT(Q)) - Area(DT(Q))) / sqrt(Area_var(DT(Q)))

    Parameters:
    -----------
    areas : np.ndarray
        Array of triangle areas from DT(Q)

    Returns:
    --------
    z_scores : np.ndarray
        Z-scores for each triangle area.
        Large/wide triangles get negative z-scores.

    Reference: Equation (2) in Kim & Cho (2019), Section 3
    """
    area_mean = np.mean(areas)
    area_std = np.std(areas)

    # Formula from paper Eq. (2): (mean - value) / sqrt(variance)
    # Large areas get negative z-scores and should be removed
    z_scores = (area_mean - areas) / (area_std + 1e-10)

    return z_scores


def z_length(lengths: np.ndarray) -> np.ndarray:
    """
    Calculate z-score normalization for edge lengths.

    According to Equation (3) in the paper:
    LengthZ = (Length_mean(DT(Q)) - Length(DT(Q))) / sqrt(Length_var(DT(Q)))

    Parameters:
    -----------
    lengths : np.ndarray
        Array of edge lengths from DT(Q)

    Returns:
    --------
    z_scores : np.ndarray
        Z-scores for each edge length.
        Long edges get negative z-scores.

    Reference: Equation (3) in Kim & Cho (2019), Section 3
    """
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)

    # Formula from paper Eq. (3): (mean - value) / sqrt(variance)
    # Long edges get negative z-scores and should be removed
    z_scores = (length_mean - lengths) / (length_std + 1e-10)

    return z_scores


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

    def __init__(self, MinPts: int = 6, area_threshold: float = -2.0, length_threshold: float = -2.0):
        """
        Initialize DTSCAN clustering algorithm.

        Parameters:
        -----------
        MinPts : int
            Minimum number of neighboring nodes required for a point to be
            considered as a core point in a cluster (default: 6)
            This is the ONLY hyperparameter of DTSCAN per the paper.
            Directly analogous to DBSCAN's MinPts parameter
            Reference: Section 3, clustering process step 2

        area_threshold : float
            Z-score threshold for filtering triangles (default: -2.0 for debugging)
            Paper doesn't specify exact value

        length_threshold : float
            Z-score threshold for filtering edges (default: -2.0 for debugging)
            Paper doesn't specify exact value
        """
        self.MinPts = MinPts  # Using paper's terminology
        self.area_threshold = area_threshold
        self.length_threshold = length_threshold

        # Storage for intermediate results and visualization
        self.triangulation = None  # DT(Q) - Delaunay triangulation
        self.graph_edges = None     # Initial graph from triangulation
        self.filtered_graph = None  # Graph after removing global effects
        self.labels = None          # Final cluster labels

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
        print("Step 1: Performing Delaunay triangulation...")
        self.triangulation = Delaunay(X)

        # Step 2: Build graph from triangulation
        print("Step 2: Building graph from triangulation...")
        self.graph_edges = self._build_graph_from_triangulation(X)

        # Step 3: Remove global effects using z-score normalization
        print("Step 3: Removing global effects (filtering outlier edges/triangles)...")
        self.filtered_graph = self._remove_global_effects(X)

        # Step 4: Apply density-based clustering mechanism
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

    def _remove_global_effects(self, X: np.ndarray) -> Dict[int, Set[int]]:
        """
        Remove edges and triangles that are outliers based on z-score normalization.

        This step filters out:
        - Triangles with unusually large areas (likely between clusters)
        - Edges with unusually long lengths (likely noise connections)

        Reference: Section 3, Equations (2) and (3)
        """
        print("   Building Triangle objects...")
        # Create Triangle objects for all simplices
        triangles = []
        for simplex in self.triangulation.simplices:
            points = X[simplex]
            triangle = Triangle(points=points, vertex_indices=simplex)
            triangles.append(triangle)

        # Calculate all triangle areas using the Triangle class
        print("   Computing triangle areas...")
        areas = np.array([tri.compute_area() for tri in triangles])

        # Apply z-score normalization according to Equation (2)
        print("   Applying z-score normalization to areas (Eq. 2)...")
        area_z_scores = z_area(areas)

        # Build edge-to-triangle mapping and calculate all edge lengths
        print("   Computing edge lengths...")
        edge_to_length = {}
        edge_to_triangles = defaultdict(list)  # Track which triangles contain each edge

        for tri_idx, triangle in enumerate(triangles):
            simplex = triangle.vertex_indices
            edge_lengths = triangle.compute_edge_lengths()

            # Map edges to their lengths and parent triangles
            edges = [
                tuple(sorted([simplex[0], simplex[1]])),
                tuple(sorted([simplex[1], simplex[2]])),
                tuple(sorted([simplex[2], simplex[0]]))
            ]

            for edge_idx, edge in enumerate(edges):
                if edge not in edge_to_length:
                    edge_to_length[edge] = edge_lengths[edge_idx]
                edge_to_triangles[edge].append(tri_idx)

        # Apply z-score normalization to edge lengths according to Equation (3)
        print("   Applying z-score normalization to edge lengths (Eq. 3)...")
        all_lengths = np.array(list(edge_to_length.values()))
        length_z_scores_array = z_length(all_lengths)

        # Map z-scores back to edges
        edge_to_z_score = {}
        for edge, z_score in zip(edge_to_length.keys(), length_z_scores_array):
            edge_to_z_score[edge] = z_score

        # Filter based on z-scores
        # According to the paper (page 5): "selecting an appropriate threshold value"
        # and "the optimal value for the data distribution characteristic"
        # The paper removes triangles/edges with negative z-scores (outliers)
        # Since the paper doesn't specify the exact threshold, we filter elements
        # with negative z-scores (below mean = outliers that are too large/long)

        # Filter triangles based on z-score threshold
        # Negative z-scores indicate triangles larger than mean (outliers between clusters)
        # Keep triangles with z-score >= threshold
        valid_triangle_indices = set()
        for tri_idx, z_score in enumerate(area_z_scores):
            if z_score >= self.area_threshold:
                valid_triangle_indices.add(tri_idx)

        print(f"   Kept {len(valid_triangle_indices)}/{len(triangles)} triangles after area filtering")

        # Rebuild graph with filtered triangles and edges
        filtered_graph = defaultdict(set)

        for tri_idx in valid_triangle_indices:
            triangle = triangles[tri_idx]
            simplex = triangle.vertex_indices

            # Get all edges of this triangle
            edges = [
                tuple(sorted([simplex[0], simplex[1]])),
                tuple(sorted([simplex[1], simplex[2]])),
                tuple(sorted([simplex[2], simplex[0]]))
            ]

            for edge in edges:
                edge_z_score = edge_to_z_score[edge]

                # Keep edges with z-score >= threshold
                # Negative z-scores indicate edges longer than mean (likely noise connections)
                if edge_z_score >= self.length_threshold:
                    v1, v2 = edge
                    filtered_graph[v1].add(v2)
                    filtered_graph[v2].add(v1)

        # Ensure all points are in the graph (even if isolated)
        for i in range(len(X)):
            if i not in filtered_graph:
                filtered_graph[i] = set()

        total_edges = sum(len(neighbors) for neighbors in filtered_graph.values()) // 2
        print(f"   Final filtered graph has {total_edges} edges")

        # Convert defaultdict to regular dict
        return dict(filtered_graph)

    def _density_based_clustering(self, n_points: int) -> np.ndarray:
        """
        Apply DBSCAN-like density-based clustering on the filtered graph.

        This implements the clustering process described in Section 3 of the paper:
        1. For each point pi, find its neighboring nodes (directly connected in graph)
        2. If |neighbors| >= MinPts, include pi in a cluster Cq
        3. Expand the cluster by searching neighboring nodes recursively
        4. Points without sufficient connections are classified as noise

        Key differences from standard DBSCAN:
        - Uses graph connectivity instead of epsilon radius
        - Neighbors are defined by edges in the filtered Delaunay graph

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
            # nodes is greater than or equal to MinPts, it is included as one cluster Cq"
            if len(neighbors) < self.MinPts:
                continue  # Point remains as noise

            # Start a new cluster Cq
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

                # If neighbor is a core point (has >= MinPts neighbors), add its neighbors to queue
                if len(neighbor_neighbors) >= self.MinPts:
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
        axes[0].triplot(X[:, 0], X[:, 1], self.triangulation.simplices, 'b-', alpha=0.3, linewidth=0.5)
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
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels[unique_labels != -1])))

        color_idx = 0
        for label in unique_labels:
            if label == -1:
                # Noise points in black
                mask = self.labels == label
                axes[2].plot(X[mask, 0], X[mask, 1], 'k.', markersize=3, alpha=0.3)
            else:
                # Cluster points in colors
                mask = self.labels == label
                axes[2].plot(X[mask, 0], X[mask, 1], '.', color=colors[color_idx], markersize=5)
                color_idx += 1

        axes[2].set_title(f'DTSCAN Clustering Result ({len(unique_labels[unique_labels != -1])} clusters)')
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()


def psr(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate PSR (Point Score Range) metric for clustering evaluation.

    PSR measures the similarity between ground truth and predicted clusters using
    the Intersection over Union (IoU) / Jaccard Index metric.

    For each true cluster j, PSR finds the best matching predicted cluster i and
    computes IoU. The final score is the average across all true clusters.

    Parameters:
    -----------
    true_labels : np.ndarray
        Ground truth cluster labels (-1 indicates noise)
    pred_labels : np.ndarray
        Predicted cluster labels (-1 indicates noise)

    Returns:
    --------
    avg_psr : float
        Average PSR score across all true clusters (range: 0-1, higher is better)

    Reference: Equation (4) in Kim & Cho (2019)
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

    # For each true cluster, find the best matching predicted cluster
    for true_cluster in unique_true:
        true_mask = true_labels_clean == true_cluster
        best_psr = 0

        for pred_cluster in unique_pred:
            pred_mask = pred_labels_clean == pred_cluster

            # Calculate intersection and union
            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)

            if union > 0:
                # IoU (Jaccard Index) as similarity measure
                psr = intersection / union
                best_psr = max(best_psr, psr)

        psr_scores.append(best_psr)

    avg_psr = np.mean(psr_scores) if psr_scores else 0.0
    return avg_psr
