import optuna as hpo
import mlflow
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from dtscan import DTSCAN
from dtc import DTC
from dtcsvm import DTCSVM
from ascdt import ASCDT
from cluster_sets import clusters_paper, load_s5

import io
import time
import logging
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors
from contextlib import redirect_stdout

mlflow.set_tracking_uri('file:../mlruns')

def choose_model(X, X_scaled, algorithm_name, search_params, trial):
    """
    Choose and configure a clustering model based on algorithm name and hyperparameters.

    Note: This function suggests parameters dynamically based on the search_params structure.
    It handles both naming conventions (e.g., 'min_pts' vs 'MinPts').

    Data usage rules:
    - DBSCAN: Always scaled (distance-based)
    - DTC, DTCSVM: Always unscaled (triangulation-based, needs original coordinates)
    - Others: Optimized as hyperparameter
    """
    # Determine which data to use based on algorithm
    if algorithm_name == 'DBSCAN':
        X_use = X_scaled  # DBSCAN always uses scaled data
    elif algorithm_name in ['DTC', 'DTCS', 'DTCSVM']:
        X_use = X  # Triangulation algorithms always use unscaled data
    else:
        # For other algorithms, optimize whether to use scaled data
        use_scaled = trial.suggest_categorical('use_scaled', [True, False])
        X_use = X_scaled if use_scaled else X

    try:
        match algorithm_name:
            case 'DTSCAN':
                params = search_params['DTSCAN']

                min_pts = trial.suggest_int('MinPts', *params['MinPts']['range'])
                area_threshold = trial.suggest_float('area_threshold', *params['area_threshold']['range'])
                length_threshold = trial.suggest_float('length_threshold', *params['length_threshold']['range'])

                f = io.StringIO()
                with redirect_stdout(f):
                    return DTSCAN(MinPts=min_pts, area_threshold=area_threshold,
                                 length_threshold=length_threshold).fit_predict(X_use)

            case 'DTC':
                params = search_params['DTC']
                if 'minPts' in params:
                    min_pts = trial.suggest_int('minPts', *params['minPts']['range'])
                else:
                    min_pts = trial.suggest_int('min_pts', *params['min_pts']['range'])

                local_std = trial.suggest_float('local_std', *params['local_std']['range'])
                kde = trial.suggest_categorical('kde', params['kde']['options']) if 'kde' in params else False

                # DTC works directly with the data array - create DataFrame properly
                # X_use is always X (unscaled) for DTC
                df = pd.DataFrame(X_use, columns=['x', 'y'])
                dtc_model = DTC(data=df, minPts=min_pts, local_std=local_std, kde=kde)

                f = io.StringIO()
                with redirect_stdout(f):
                    result_df = dtc_model.tri_dbscan()

                if result_df is None or 'est_clust' not in result_df.columns:
                    print(f"Error: DTC failed to cluster data")
                    return np.array([-1] * len(X_use))

                # Result has columns: x, y, index, est_clust
                # Need to sort by 'index' to match original order
                result_df = result_df.sort_values('index').reset_index(drop=True)
                labels = result_df['est_clust'].values

                # Convert 0 (noise in DTC) to -1 (standard noise label)
                return np.where(labels == 0, -1, labels)

            case 'DTCS' | 'DTCSVM':
                params = search_params.get('DTCSVM', search_params.get('DTCS', {}))
                if 'minPts' in params:
                    min_pts = trial.suggest_int('minPts', *params['minPts']['range'])
                else:
                    min_pts = trial.suggest_int('min_pts', *params['min_pts']['range'])

                local_std = trial.suggest_float('local_std', *params['local_std']['range'])
                svm_c = trial.suggest_float('svm_c', *params['svm_c']['range'], log=params['svm_c'].get('log', False))
                svm_gamma = trial.suggest_categorical('svm_gamma', params['svm_gamma']['options'])
                kde = trial.suggest_categorical('kde', params['kde']['options']) if 'kde' in params else False

                # DTCSVM needs DataFrame - ensure it's properly formatted
                # X_use is always X (unscaled) for DTCSVM
                df = pd.DataFrame(X_use, columns=['x', 'y'])
                df = df.reset_index(drop=True)  # Clean RangeIndex starting from 0

                dtcs_model = DTCSVM(data=df, minPts=min_pts, local_std=local_std,
                                   svm_c=svm_c, svm_gamma=svm_gamma, kde=kde)

                f = io.StringIO()
                with redirect_stdout(f):
                    result_df = dtcs_model.tri_dbscan()

                if result_df is None or 'est_clust' not in result_df.columns:
                    print(f"Error: DTCSVM failed to cluster data")
                    return np.array([-1] * len(X_use))

                # Result has columns: x, y, index, est_clust
                # Need to sort by 'index' to match original order
                result_df = result_df.sort_values('index').reset_index(drop=True)
                labels = result_df['est_clust'].values

                # Convert 0 (noise in DTCSVM) to -1 (standard noise label)
                return np.where(labels == 0, -1, labels)

            case 'ASCDT':
                params = search_params['ASCDT']
                min_cluster_size = trial.suggest_int('min_cluster_size', *params['min_cluster_size']['range'])
                beta = trial.suggest_float('beta', *params['beta']['range'])
                return ASCDT(min_cluster_size=min_cluster_size, beta=beta).fit_predict(X_use)

            case 'DBSCAN':
                params = search_params['DBSCAN']
                eps = trial.suggest_float('eps', *params['eps']['range'])
                min_samples = trial.suggest_int('min_samples', *params['min_samples']['range'])
                # DBSCAN always uses scaled data (X_use is X_scaled for DBSCAN)
                return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_use)

            case 'HDBSCAN':
                params = search_params['HDBSCAN']
                min_cluster_size = trial.suggest_int('min_cluster_size', *params['min_cluster_size']['range'])
                alpha = trial.suggest_float('alpha', *params['alpha']['range'])
                return HDBSCAN(min_cluster_size=min_cluster_size, alpha=alpha).fit_predict(X_use)

            case 'KMeans':
                params = search_params['KMeans']
                n_clusters = trial.suggest_int('n_clusters', *params['n_clusters']['range'])
                n_init = trial.suggest_categorical('n_init', params['n_init']['options'])
                return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42).fit_predict(X_use)

            case 'Spectral':
                params = search_params['Spectral']
                n_clusters = trial.suggest_int('n_clusters', *params['n_clusters']['range'])
                n_neighbors = trial.suggest_int('n_neighbors', *params['n_neighbors']['range'])
                return SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbors,
                                         random_state=42).fit_predict(X_use)

            case 'GMM':
                params = search_params['GMM']
                n_components = trial.suggest_int('n_components', *params['n_components']['range'])
                covariance_type = trial.suggest_categorical('covariance_type', params['covariance_type']['options'])
                n_init = trial.suggest_categorical('n_init', params['n_init']['options'])
                return GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                      n_init=n_init, random_state=42).fit_predict(X_use)

            case _:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

    except Exception as e:
        print(f"Error choosing model: {algorithm_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([-1] * len(X))



def optimize_single_task(task_params):
    """
    Optimize a single (dataset, algorithm) combination.

    This function runs in a separate process for parallel optimization.

    Parameters
    ----------
    task_params : tuple
        (dataset_name, algorithm, X, true_labels, n_trials, experiment_name,
         hyperparameter_ranges, mlflow_uri, metric_func, use_optuna_multiprocessing, n_jobs)

    Returns
    -------
    dict
        Optimization results including best parameters and scores
    """
    # Suppress logging in worker processes
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    (dataset_name, algorithm, X, true_labels, n_trials,
     experiment_name, hyperparameter_ranges, mlflow_uri, metric_func,
     use_optuna_multiprocessing, optuna_n_jobs) = task_params

    # Prepare data
    expected_clusters = len(np.unique(true_labels[true_labels != -1]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Get algorithm-specific parameters
    params = hyperparameter_ranges[algorithm]


    # Define objective function for this task
    def objective(trial):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trial_num = trial.number + 1

        try:
            print(f"[{timestamp}] Dataset: {dataset_name}, Algorithm: {algorithm}, Trial: {trial_num}/{n_trials}")

            # Use the existing choose_model function
            labels = choose_model(X, X_scaled, algorithm, hyperparameter_ranges, trial)

            # Calculate metric score
            # Check if this is a supervised metric (has true labels) or unsupervised (uses X)
            import inspect
            sig = inspect.signature(metric_func)
            n_params = len(sig.parameters)

            if n_params == 2:
                # Could be (true_labels, pred_labels) or (X, labels)
                # Try with X first (for unsupervised metrics like GGDS)
                param_names = list(sig.parameters.keys())
                if 'points' in param_names or 'X' in param_names or 'X_data' in param_names:
                    score = metric_func(X, labels)
                else:
                    # Assume supervised metric (true_labels, pred_labels)
                    score = metric_func(true_labels, labels)
            else:
                # Default: try supervised first, then unsupervised
                try:
                    score = metric_func(true_labels, labels)
                except:
                    score = metric_func(X, labels)

            # Store number of clusters found
            n_clusters = len(np.unique(labels[labels != -1]))
            trial.set_user_attr('n_clusters', n_clusters)

            return score

        except Exception as e:
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp_error}] ERROR - Dataset: {dataset_name}, Algorithm: {algorithm}, Trial: {trial_num}/{n_trials}")
            print(f"    Error details: {str(e)}")
            return 0.0

    # Setup study
    study_name = f"{dataset_name}_{algorithm}"
    study = hpo.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=hpo.samplers.TPESampler(seed=42)
    )

    # Optimize
    start_time = time.time()
    if use_optuna_multiprocessing:
        # Use Optuna's built-in multiprocessing
        study.optimize(objective, n_trials=n_trials, n_jobs=optuna_n_jobs, show_progress_bar=False)
    else:
        # Sequential optimization (parallelization happens at task level)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    duration = time.time() - start_time

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=study_name):
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("expected_clusters", expected_clusters)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_score", study.best_value)
        mlflow.log_metric("optimization_time_seconds", duration)

        if 'n_clusters' in study.best_trial.user_attrs:
            mlflow.log_metric("best_n_clusters",
                            study.best_trial.user_attrs['n_clusters'])

    # Return results
    result = {
        'dataset': dataset_name,
        'algorithm': algorithm,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'expected_clusters': expected_clusters,
        'n_clusters': study.best_trial.user_attrs.get('n_clusters', None),
        'duration': duration
    }

    return result


def run_parallel_optimization(datasets, algorithms, metric_func, hyperparameter_ranges,
                              n_trials=100, experiment_name="HPO_experiment",
                              n_jobs=-1, mlflow_uri="file:../mlruns", verbose=True,
                              use_optuna_multiprocessing=False):
    """
    Run parallel hyperparameter optimization across multiple datasets and algorithms.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to (X, labels) tuples
    algorithms : list
        List of algorithm names to optimize
    metric_func : callable
        Metric function with signature: metric_func(true_labels, predicted_labels) -> float
        Should return higher values for better performance
    hyperparameter_ranges : dict
        Dictionary defining hyperparameter search spaces for each algorithm
    n_trials : int, default=100
        Number of optimization trials per (dataset, algorithm) combination
    experiment_name : str, default="HPO_experiment"
        Name for the MLflow experiment
    n_jobs : int, default=-1
        Number of parallel processes (-1 uses all CPU cores)
    mlflow_uri : str, default="file:../mlruns"
        MLflow tracking URI
    verbose : bool, default=True
        Whether to print progress messages
    use_optuna_multiprocessing : bool, default=False
        If True, uses Optuna's built-in multiprocessing (parallelizes trials within each study).
        If False, uses Pool-based parallelization (parallelizes across dataset-algorithm combinations).

    Returns
    -------
    dict
        Nested dictionary of results: {dataset_name: {algorithm_name: result_dict}}
    """
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Determine number of parallel jobs
    n_cores = cpu_count()
    n_jobs = n_cores if n_jobs == -1 else n_jobs

    # Create all tasks
    tasks = []
    for dataset_name, (X, labels) in datasets.items():
        for algorithm in algorithms:
            task = (dataset_name, algorithm, X, labels, n_trials,
                   experiment_name, hyperparameter_ranges, mlflow_uri, metric_func,
                   use_optuna_multiprocessing, n_jobs)
            tasks.append(task)

    if verbose:
        print("\n" + "="*80)
        print("STARTING PARALLEL OPTIMIZATION")
        print("="*80)
        print(f"Datasets: {list(datasets.keys())}")
        print(f"Algorithms: {algorithms}")
        print(f"Trials per task: {n_trials}")
        if use_optuna_multiprocessing:
            print(f"Parallelization: Optuna (trials within each study)")
            print(f"Trials run in parallel: {n_jobs}")
        else:
            print(f"Parallelization: Pool (across dataset-algorithm combinations)")
            print(f"Parallel processes: {n_jobs}")
        print(f"Total tasks: {len(tasks)}")
        print("="*80 + "\n")
    else:
        # Suppress all logging if not verbose
        logging.getLogger("optuna").setLevel(logging.CRITICAL)
        logging.getLogger("mlflow").setLevel(logging.CRITICAL)

    # Suppress Optuna and MLflow logging
    hpo_logger = logging.getLogger("optuna")
    hpo_logger.setLevel(logging.WARNING)
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.WARNING)

    # Run optimization
    start_time = time.time()
    results = []
    completed = 0
    total_tasks = len(tasks)

    if use_optuna_multiprocessing or n_jobs == 1:
        # Sequential execution of tasks (Optuna handles parallelization internally)
        for task in tasks:
            result = optimize_single_task(task)
            results.append(result)
            completed += 1
    else:
        # Parallel execution across tasks using Pool
        with Pool(processes=n_jobs) as pool:
            for result in pool.imap(optimize_single_task, tasks):
                results.append(result)
                completed += 1

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average time per task: {total_time/len(tasks):.1f} seconds")
        print("="*80)

    # Organize results by dataset and algorithm
    organized_results = {}
    for result in results:
        dataset = result['dataset']
        algorithm = result['algorithm']

        if dataset not in organized_results:
            organized_results[dataset] = {}

        organized_results[dataset][algorithm] = result

    return organized_results


def results_to_dataframe(results):
    """
    Convert optimization results to a pandas DataFrame.

    Parameters
    ----------
    results : dict
        Nested dictionary from run_parallel_optimization

    Returns
    -------
    pd.DataFrame
        Results formatted as a DataFrame
    """
    results_data = []
    for dataset_name, algo_results in results.items():
        for algo_name, result in algo_results.items():
            row = {
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Score': result['best_score'],
                'N Clusters': result.get('n_clusters', 'N/A'),
                'Expected': result['expected_clusters'],
                'Time (s)': f"{result['duration']:.1f}",
                'Best Params': str(result['best_params'])
            }
            results_data.append(row)

    df = pd.DataFrame(results_data)
    df = df.sort_values(['Dataset', 'Score'], ascending=[True, False])
    return df

