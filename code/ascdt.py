import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from collections import defaultdict

class ASCDT:
    """
    Adaptive Spatial Clustering algorithm based on Delaunay Triangulation (ASCDT)
    
    This algorithm can automatically discover clusters of complicated shapes and 
    non-homogeneous densities in a spatial database without needing parameters.
    """
    
    def __init__(self, min_cluster_size=5, beta=1.0):
        """
        Initialize ASCDT clustering algorithm
        
        Parameters:
        -----------
        min_cluster_size : int, default=5
            Minimum number of points to form a cluster
        beta : float, default=1.0
            Control factor for local cut-off value sensitivity (1.0 to 1.5)
        """
        self.min_cluster_size = min_cluster_size
        self.beta = beta
        self.labels_ = None
        self.n_clusters_ = 0
        
    def fit(self, X):
        """
        Perform ASCDT clustering on spatial data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            2D spatial point data
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.array(X)
        if X.shape[1] != 2:
            raise ValueError("ASCDT only works with 2D spatial data")
            
        # Step 1: Construct Delaunay triangulation
        tri = self._build_delaunay(X)
        
        # Step 2: Remove global long edges
        sub_graphs = self._global_cut(X, tri)
        
        # Step 3: Process each sub-graph to remove local inconsistent edges
        clusters = []
        for sub_graph in sub_graphs:
            if len(sub_graph['points']) < self.min_cluster_size:
                continue
            local_clusters = self._local_cut(X, sub_graph)
            clusters.extend(local_clusters)
        
        # Assign cluster labels
        self._assign_labels(X, clusters)
        
        return self
        
    def fit_predict(self, X):
        """
        Perform clustering and return cluster labels
        """
        self.fit(X)
        return self.labels_
        
    def _build_delaunay(self, X):
        """
        Build Delaunay triangulation and extract edges
        """
        # Handle duplicate points by adding small noise
        X_unique = X + np.random.normal(0, 1e-10, X.shape)
        
        tri = Delaunay(X_unique)
        
        # Extract all edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edges.add(edge)
        
        # Calculate edge lengths
        edge_list = []
        for i, j in edges:
            length = euclidean(X[i], X[j])
            edge_list.append({'start': i, 'end': j, 'length': length})
            
        return {'triangulation': tri, 'edges': edge_list, 'points': X}
        
    def _global_cut(self, X, tri_data):
        """
        Remove global long edges based on statistical features
        """
        edges = tri_data['edges']
        n_points = len(X)
        
        # Calculate global statistics
        lengths = [e['length'] for e in edges]
        global_mean = np.mean(lengths)
        global_std = np.std(lengths)
        
        # Build adjacency list
        adj_list = defaultdict(list)
        for edge in edges:
            adj_list[edge['start']].append(edge)
            adj_list[edge['end']].append({
                'start': edge['end'], 
                'end': edge['start'], 
                'length': edge['length']
            })
        
        # Mark edges for removal based on global cut value
        edges_to_keep = []
        for point_id in range(n_points):
            if point_id not in adj_list:
                continue
                
            adjacent_edges = adj_list[point_id]
            if len(adjacent_edges) == 0:
                continue
                
            # Calculate local mean for this point
            local_lengths = [e['length'] for e in adjacent_edges]
            local_mean = np.mean(local_lengths)
            
            # Adaptive global cut value
            if local_mean > 0:
                adaptive_factor = global_mean / local_mean
                global_cut_value = global_mean + adaptive_factor * global_std
            else:
                global_cut_value = global_mean + global_std
            
            # Keep edges shorter than cut value
            for edge in adjacent_edges:
                if edge['length'] < global_cut_value:
                    edge_tuple = tuple(sorted([edge['start'], edge['end']]))
                    edges_to_keep.append(edge_tuple)
        
        # Remove duplicates
        edges_to_keep = list(set(edges_to_keep))
        
        # Find connected components
        components = self._find_connected_components(n_points, edges_to_keep)
        
        # Create sub-graphs
        sub_graphs = []
        for component in components:
            if len(component) >= self.min_cluster_size:
                component_edges = []
                for i, j in edges_to_keep:
                    if i in component and j in component:
                        length = euclidean(X[i], X[j])
                        component_edges.append({
                            'start': i, 'end': j, 'length': length
                        })
                sub_graphs.append({
                    'points': list(component),
                    'edges': component_edges
                })
        
        return sub_graphs
        
    def _local_cut(self, X, sub_graph):
        """
        Remove local inconsistent edges (long edges, chains, necks)
        """
        points = sub_graph['points']
        edges = sub_graph['edges']
        
        if len(edges) == 0:
            return [points]
        
        # Build adjacency for sub-graph
        adj_list = defaultdict(list)
        for edge in edges:
            adj_list[edge['start']].append(edge)
            adj_list[edge['end']].append({
                'start': edge['end'],
                'end': edge['start'],
                'length': edge['length']
            })
        
        # Calculate mean variation for sub-graph
        point_variations = []
        for point_id in points:
            if point_id in adj_list:
                adjacent = adj_list[point_id]
                if len(adjacent) > 1:
                    lengths = [e['length'] for e in adjacent]
                    point_variations.append(np.std(lengths))
        
        mean_variation = np.mean(point_variations) if point_variations else 0
        
        # Remove local long edges
        edges_to_keep = []
        for point_id in points:
            if point_id not in adj_list:
                continue
                
            # Get 2-order neighbors
            two_order_neighbors = self._get_k_order_neighbors(adj_list, point_id, k=2)
            
            if len(two_order_neighbors) > 0:
                # Calculate local cut value
                neighbor_edges = []
                for n in two_order_neighbors:
                    if n in adj_list:
                        neighbor_edges.extend(adj_list[n])
                
                if neighbor_edges:
                    mean_2_order = np.mean([e['length'] for e in neighbor_edges])
                    local_cut_value = mean_2_order + self.beta * mean_variation
                    
                    # Keep edges below cut value
                    for edge in adj_list[point_id]:
                        if edge['length'] < local_cut_value:
                            edge_tuple = tuple(sorted([edge['start'], edge['end']]))
                            edges_to_keep.append(edge_tuple)
        
        # Remove duplicates
        edges_to_keep = list(set(edges_to_keep))
        
        # Apply chain and neck removal
        edges_to_keep = self._remove_chains_and_necks(X, points, edges_to_keep)
        
        # Find final connected components
        components = self._find_connected_components(max(points)+1, edges_to_keep)
        
        # Filter small clusters
        clusters = []
        for component in components:
            component_in_subgraph = [p for p in component if p in points]
            if len(component_in_subgraph) >= self.min_cluster_size:
                clusters.append(component_in_subgraph)
        
        return clusters
    
    def _remove_chains_and_necks(self, X, points, edges):
        """
        Remove chain edges and neck connections using local aggregation force
        """
        if len(edges) == 0:
            return edges
            
        # Build adjacency
        adj_list = defaultdict(set)
        for i, j in edges:
            adj_list[i].add(j)
            adj_list[j].add(i)
        
        edges_to_remove = set()
        
        # Remove chains (vertices with degree 2 in a line)
        for point in points:
            if len(adj_list[point]) == 2:
                neighbors = list(adj_list[point])
                # Check if this forms a chain
                if neighbors[0] not in adj_list[neighbors[1]]:
                    # This is a chain point, remove one edge
                    edge_to_remove = tuple(sorted([point, neighbors[0]]))
                    edges_to_remove.add(edge_to_remove)
        
        # Calculate local aggregation forces for neck detection
        for point in points:
            if point not in adj_list or len(adj_list[point]) == 0:
                continue
                
            # Get 2-order neighbors
            two_order = self._get_k_order_neighbors_from_edges(edges, point, k=2)
            
            if len(two_order) < 3:  # Need enough neighbors
                continue
                
            # Calculate cohesive force
            cohesive_force = np.zeros(2)
            for neighbor in two_order:
                if neighbor != point:
                    direction = X[neighbor] - X[point]
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        force = direction / (distance ** 2)
                        cohesive_force += force
            
            # Check each adjacent edge
            for neighbor in list(adj_list[point]):
                direction = X[neighbor] - X[point]
                distance = np.linalg.norm(direction)
                if distance > 0:
                    edge_force = direction / (distance ** 2)
                    
                    # Calculate angle between forces
                    cos_angle = np.dot(cohesive_force, edge_force) / (
                        np.linalg.norm(cohesive_force) * np.linalg.norm(edge_force) + 1e-10
                    )
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    # Remove edge if angle > 90 degrees (neck or wrong direction)
                    if angle > np.pi / 2:
                        edge_to_remove = tuple(sorted([point, neighbor]))
                        edges_to_remove.add(edge_to_remove)
        
        # Remove marked edges
        final_edges = [e for e in edges if e not in edges_to_remove]
        
        return final_edges
    
    def _get_k_order_neighbors(self, adj_list, point, k=2):
        """
        Get k-order neighbors of a point
        """
        visited = set([point])
        current_level = set([point])
        
        for _ in range(k):
            next_level = set()
            for p in current_level:
                if p in adj_list:
                    for edge in adj_list[p]:
                        neighbor = edge['end']
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
            current_level = next_level
            
        return list(visited)
    
    def _get_k_order_neighbors_from_edges(self, edges, point, k=2):
        """
        Get k-order neighbors from edge list
        """
        adj = defaultdict(set)
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
            
        visited = set([point])
        current_level = set([point])
        
        for _ in range(k):
            next_level = set()
            for p in current_level:
                for neighbor in adj[p]:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current_level = next_level
            
        return list(visited)
    
    def _find_connected_components(self, n_points, edges):
        """
        Find connected components from edges
        """
        # Build adjacency list
        adj_list = defaultdict(set)
        for i, j in edges:
            adj_list[i].add(j)
            adj_list[j].add(i)
        
        # Find components using DFS
        visited = set()
        components = []
        
        for point in range(n_points):
            if point not in visited and point in adj_list:
                component = []
                stack = [point]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        for neighbor in adj_list[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if len(component) >= self.min_cluster_size:
                    components.append(component)
        
        return components
    
    def _assign_labels(self, X, clusters):
        """
        Assign cluster labels to points
        """
        n_points = len(X)
        self.labels_ = np.full(n_points, -1)  # -1 for noise
        
        for cluster_id, cluster_points in enumerate(clusters):
            for point_id in cluster_points:
                self.labels_[point_id] = cluster_id
        
        self.n_clusters_ = len(clusters)