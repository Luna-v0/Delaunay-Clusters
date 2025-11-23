# DTSCAN: Delaunay Triangulation-Based Spatial Clustering

Implementation of the DTSCAN algorithm from:

**"Delaunay Triangulation-Based Spatial Clustering Technique for Enhanced Adjacent Boundary Detection and Segmentation of LiDAR 3D Point Clouds"**
by Jongwon Kim and Jeongho Cho (2019)

## Project Structure

```
dtscan/
├── code/
│   ├── dtscan.py           # DTSCAN algorithm implementation
│   └── cluster_sets.py     # Dataset loading utilities
│
└── notebooks/
    └── hyperparameter_optimization.ipynb  # Main HPO notebook (USE THIS!)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Open the hyperparameter optimization notebook
jupyter notebook notebooks/hyperparameter_optimization.ipynb
```

## Hyperparameter Optimization

The main notebook **`notebooks/hyperparameter_optimization.ipynb`** provides:

- ✅ **PSR score optimization** (primary metric)
- ✅ **500 trials per algorithm** (configurable)
- ✅ **Independent optimization** for each dataset (S1, S2, S3)
- ✅ **Small, focused cells** for easy execution
- ✅ **MLflow experiment tracking**
- ✅ **Automatic result export** to JSON

### Algorithms Optimized

For each dataset, the notebook finds best hyperparameters for:

1. **DTSCAN** - Delaunay Triangulation-based clustering
2. **DBSCAN** - Density-based clustering
3. **KMeans** - Centroid-based clustering
4. **Spectral** - Graph-based clustering
5. **GMM** - Gaussian Mixture Model

### Configuration

Edit the first code cell:

```python
N_TRIALS = 500  # Increase for better results, decrease for faster execution
```

### View Results

After optimization completes:

1. **Summary table** in notebook
2. **JSON export** (`best_params_psr.json`)
3. **MLflow UI**: `mlflow ui --backend-store-uri file:../mlruns`

### Using Optimized Parameters

```python
import json
from dtscan import DTSCAN

# Load best parameters
with open('best_params_psr.json', 'r') as f:
    params = json.load(f)

# Use optimized parameters for S1
s1_params = params['S1']['DTSCAN']['best_params']
dtscan = DTSCAN(**s1_params)
labels = dtscan.fit_predict(X)
```

## Key Features

### Independent Per-Dataset Optimization

Each dataset is optimized separately:
- **S1**: 788 points, 7 clusters
- **S2**: 300 points, 3 clusters
- **S3**: 373 points, 2 clusters

### Intelligent Search with Optuna

- **TPE algorithm**: Learns from previous trials
- **Smart sampling**: Focuses on promising regions
- **Efficiency**: 500 trials often beats exhaustive grid search

### Automatic Experiment Tracking

All trials logged to MLflow with:
- Parameters tested
- PSR scores achieved
- Cluster counts
- Optimization history

## Citation

```bibtex
@article{kim2019dtscan,
  title={Delaunay Triangulation-Based Spatial Clustering Technique for Enhanced Adjacent Boundary Detection and Segmentation of LiDAR 3D Point Clouds},
  author={Kim, Jongwon and Cho, Jeongho},
  journal={Sensors},
  year={2019}
}
```

