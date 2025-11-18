import numpy as np
import pandas as pd
from glob import glob

def cluster_test_1():
    """
    Generate synthetic test data similar to S1 in the paper.
    Multiple clusters with different shapes and densities, including touching clusters.

    Reference: Figure 6a in the paper
    """
    np.random.seed(42)

    # Create multiple clusters with different characteristics
    clusters = []

    # Cluster 1: Dense circular cluster
    theta = np.random.uniform(0, 2*np.pi, 150)
    r = np.random.normal(0, 0.1, 150)
    x1 = r * np.cos(theta) - 1.5
    y1 = r * np.sin(theta) + 1
    clusters.append(np.column_stack([x1, y1]))

    # Cluster 2: Elongated cluster
    x2 = np.random.normal(0, 0.3, 100)
    y2 = np.random.normal(0, 0.1, 100) + 0.8
    clusters.append(np.column_stack([x2, y2]))

    # Cluster 3: Crescent shape
    theta = np.linspace(0, np.pi, 100)
    r = 0.8 + np.random.normal(0, 0.05, 100)
    x3 = r * np.cos(theta) + 1
    y3 = r * np.sin(theta)
    clusters.append(np.column_stack([x3, y3]))

    # Cluster 4: Small dense cluster (touching cluster 3)
    x4 = np.random.normal(1.8, 0.08, 50)
    y4 = np.random.normal(0.3, 0.08, 50)
    clusters.append(np.column_stack([x4, y4]))

    # Cluster 5: Sparse cluster
    x5 = np.random.uniform(-1, 0, 80)
    y5 = np.random.uniform(-0.5, 0.5, 80)
    clusters.append(np.column_stack([x5, y5]))

    # Cluster 6: Another circular cluster
    theta = np.random.uniform(0, 2*np.pi, 120)
    r = np.random.normal(0, 0.15, 120)
    x6 = r * np.cos(theta) + 1.2
    y6 = r * np.sin(theta) - 1
    clusters.append(np.column_stack([x6, y6]))

    # Cluster 7: Small cluster with chain connection
    x7 = np.random.normal(-1.5, 0.1, 60)
    y7 = np.random.normal(-0.8, 0.1, 60)
    clusters.append(np.column_stack([x7, y7]))

    # Add some chain noise connecting clusters
    x_chain = np.linspace(-1.5, -1, 5)
    y_chain = np.linspace(-0.6, 0, 5) + np.random.normal(0, 0.02, 5)
    chain_noise = np.column_stack([x_chain, y_chain])

    # Add some background noise
    x_noise = np.random.uniform(-2, 2.5, 20)
    y_noise = np.random.uniform(-1.5, 1.5, 20)
    background_noise = np.column_stack([x_noise, y_noise])

    # Combine all data
    X = np.vstack(clusters + [chain_noise, background_noise])

    # Create true labels for evaluation
    true_labels = []
    for i, cluster in enumerate(clusters):
        true_labels.extend([i] * len(cluster))
    true_labels.extend([-1] * len(chain_noise))  # Chain noise as noise
    true_labels.extend([-1] * len(background_noise))  # Background noise

    return X, np.array(true_labels)


def cluster_test_2():
    """
    Generate synthetic test data similar to S2 in the paper.
    Concentric/nested clusters with different densities.

    Reference: Figure 7a in the paper
    """
    np.random.seed(42)

    clusters = []

    # Outer ring cluster
    theta = np.random.uniform(0, 2*np.pi, 300)
    r = np.random.normal(1.5, 0.1, 300)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    clusters.append(np.column_stack([x1, y1]))

    # Inner dense cluster
    theta = np.random.uniform(0, 2*np.pi, 200)
    r = np.random.normal(0, 0.2, 200)
    x2 = r * np.cos(theta) + 0.3
    y2 = r * np.sin(theta)
    clusters.append(np.column_stack([x2, y2]))

    # Small cluster inside
    x3 = np.random.normal(-0.5, 0.1, 100)
    y3 = np.random.normal(0.5, 0.1, 100)
    clusters.append(np.column_stack([x3, y3]))

    X = np.vstack(clusters)

    # Create true labels
    true_labels = []
    for i, cluster in enumerate(clusters):
        true_labels.extend([i] * len(cluster))

    return X, np.array(true_labels)


def cluster_test_3():
    """
    Generate synthetic test data similar to S3 in the paper.
    Two adjacent nonlinear clusters with uneven density.

    Reference: Figure 8a in the paper
    """
    np.random.seed(42)

    clusters = []

    # First cluster: U-shaped with varying density
    # Dense part at bottom
    x1_bottom = np.random.uniform(-1, 1, 200)
    y1_bottom = np.random.normal(-0.5, 0.1, 200)

    # Sparse parts at sides
    y1_left = np.random.uniform(-0.5, 0.5, 50)
    x1_left = np.random.normal(-1, 0.1, 50)

    y1_right = np.random.uniform(-0.5, 0.5, 50)
    x1_right = np.random.normal(1, 0.1, 50)

    cluster1 = np.vstack([
        np.column_stack([x1_bottom, y1_bottom]),
        np.column_stack([x1_left, y1_left]),
        np.column_stack([x1_right, y1_right])
    ])
    clusters.append(cluster1)

    # Second cluster: Inverted U-shaped, adjacent to first
    # Sparse part at top
    x2_top = np.random.uniform(-0.8, 0.8, 80)
    y2_top = np.random.normal(0.8, 0.15, 80)

    # Denser parts at sides
    y2_left = np.random.uniform(0.3, 0.8, 100)
    x2_left = np.random.normal(-0.8, 0.08, 100)

    y2_right = np.random.uniform(0.3, 0.8, 100)
    x2_right = np.random.normal(0.8, 0.08, 100)

    cluster2 = np.vstack([
        np.column_stack([x2_top, y2_top]),
        np.column_stack([x2_left, y2_left]),
        np.column_stack([x2_right, y2_right])
    ])
    clusters.append(cluster2)

    X = np.vstack(clusters)

    # Create true labels
    true_labels = []
    for i, cluster in enumerate(clusters):
        true_labels.extend([i] * len(cluster))

    return X, np.array(true_labels)

def clusters_paper():
    """
    
    """
    files = glob('datasets/*.csv')
    cluster_sets = []
    for file in files:
        df = pd.read_csv(file)
        X = df[['x', 'y']].values
        cluster_sets.append(X)
    return cluster_sets
    
    