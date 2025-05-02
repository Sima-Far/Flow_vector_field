# -*- coding: utf-8 -*-
"""
Functions for calculating cosine similarity, performing clustering (KMedoids, KMeans),
fitting sinusoidal curves, and plotting cluster results for time series data.
(Cleaned version: No try/except or prints)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import silhouette_score
# Assume KMedoids is available, e.g., from sklearn.cluster or sklearn_extra
from sklearn.cluster import KMedoids # Or from sklearn_extra.cluster
from sklearn.cluster import KMeans
from tslearn.metrics import dtw
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from collections import OrderedDict

# --- Constants and Configuration ---
SAVGOL_WINDOW = 7
SAVGOL_ORDER = 3
DEFAULT_N_CLUSTERS_INITIAL = 5
RANDOM_STATE = 42

CLUSTER_NAMES_INITIAL = {
    0: "Static Cities", 1: "Patterned Dynamic Cities", 2: "Chaotic Cities", 3: 'A', 4: 'B'
}
CLUSTER_COLORS_INITIAL = {
    0: "#0072b2", 1: "#009e73", 2: "#e69f00", 3: '#d55e00', 4: '#cc79a7'
}
REMAPPED_CLUSTER_COLORS = {
    0: "#009e73", 1: '#d55e00', 2: "#e69f00", 3: "#0072b2",
}
REMAPPED_LABELS = {
    'cluster 1': "Slightly dynamic" + r" ($\theta<36^\circ$)", # Index 0
    'cluster 2': "Highly dynamic" + r" ($\theta<78^\circ$)",   # Index 1
    'cluster 3': "Moderate dynamic" + r" ($\theta<66^\circ$)",# Index 2
    'cluster 4': 'Static'                                     # Index 3
}
DESIRED_LEGEND_ORDER = [
    'Static', "Slightly dynamic" + r" ($\theta<36^\circ$)",
              "Moderate dynamic" + r" ($\theta<66^\circ$)", "Highly dynamic" + r" ($\theta<78^\circ$)"
]

# --- Helper Functions ---

def sinusoidal(t, A, B, C, D):
    """Sinusoidal function for curve fitting."""
    return A * np.sin(B * t + C) + D

def fit_sinusoidal(x, y):
    """Fits a sinusoidal function to time-series data. (Cleaned)"""
    amplitude_guess = np.ptp(y) / 2 if np.ptp(y) > 1e-9 else 0.1
    mean_guess = np.mean(y)
    guess = [amplitude_guess, 2 * np.pi / 12, 0, mean_guess]
    # curve_fit will raise RuntimeError or ValueError on failure
    popt, pcov = curve_fit(sinusoidal, x, y, p0=guess, maxfev=10000)
    # Optionally check pcov for finite values if needed, but no print/try
    return popt

# --- Core Logic Functions ---

def calculate_cosine_sim_mc(vector_mc):
    """Calculate cosine similarity between consecutive time steps. (Cleaned)"""
    if vector_mc.ndim != 3 or vector_mc.shape[1] != 2:
        raise ValueError(f"Input vector_mc must have shape (n_mc, 2, total_timesteps), but got {vector_mc.shape}")

    n_mc, _, total_timesteps = vector_mc.shape
    if total_timesteps < 2:
        return np.empty((n_mc, 0), dtype=float)

    v_curr = vector_mc[:, :, :-1]
    v_next = vector_mc[:, :, 1:]
    f_curr = np.linalg.norm(v_curr, axis=1)
    f_next = np.linalg.norm(v_next, axis=1)
    dot_p = np.einsum('ijk,ijk->ik', v_curr, v_next)
    denominator = f_curr * f_next
    vector_mc_cosine_sim = np.zeros_like(denominator)
    valid_mask = denominator > 1e-9
    vector_mc_cosine_sim[valid_mask] = dot_p[valid_mask] / denominator[valid_mask]
    vector_mc_cosine_sim = np.clip(vector_mc_cosine_sim, -1.0, 1.0)
    return vector_mc_cosine_sim

def calculate_dtw_distance_matrix(time_series_data):
    """Computes the DTW distance matrix. (Cleaned)"""
    if time_series_data.ndim != 2:
        raise ValueError(f"Input time_series_data must have shape (n_series, n_timesteps), but got {time_series_data.shape}")

    n_series = time_series_data.shape[0]
    distance_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(i + 1, n_series):
            ts_i = np.ascontiguousarray(time_series_data[i], dtype=np.float64)
            ts_j = np.ascontiguousarray(time_series_data[j], dtype=np.float64)
            dist = dtw(ts_i, ts_j)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

def perform_kmedoids_clustering(distance_matrix, n_clusters, random_state=RANDOM_STATE):
    """Performs KMedoids clustering. (Cleaned)"""
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"Distance matrix must be square, but got shape {distance_matrix.shape}")
    if n_clusters > distance_matrix.shape[0]:
        raise ValueError(f"Number of clusters ({n_clusters}) cannot be greater than the number of data points ({distance_matrix.shape[0]})")

    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=random_state, init='k-medoids++', max_iter=300)
    clusters = kmedoids.fit_predict(distance_matrix)
    return clusters, kmedoids

def calculate_cluster_means(time_series_data, clusters, n_clusters):
    """Calculates the mean time series for each cluster. (Cleaned)"""
    if time_series_data.ndim != 2 or clusters.ndim != 1 or time_series_data.shape[0] != len(clusters):
        # Raise error instead of returning None
        raise ValueError("Invalid input shapes for calculate_cluster_means.")

    n_series, n_timesteps = time_series_data.shape
    unique_labels = np.unique(clusters)
    max_label = np.max(unique_labels) if len(unique_labels) > 0 else -1
    effective_n_clusters = max(n_clusters, max_label + 1)

    cluster_means = np.zeros((effective_n_clusters, n_timesteps))
    cluster_counts = np.zeros(effective_n_clusters, dtype=int)

    np.add.at(cluster_means, clusters, time_series_data)
    cluster_counts = np.bincount(clusters, minlength=effective_n_clusters)

    for c in range(effective_n_clusters):
        if cluster_counts[c] > 0:
            cluster_means[c] /= cluster_counts[c]
        # No warning for empty clusters

    return cluster_means[:n_clusters]

def remap_clusters_and_recalculate_means(clusters_original, time_series_data, remap_rules, final_mapping_dict):
    """Remaps cluster labels and recalculates means. (Cleaned)"""
    clusters_intermediate = clusters_original.copy()
    for old_label, new_label in remap_rules:
        clusters_intermediate[clusters_intermediate == old_label] = new_label

    # Vectorize will raise error if mapping fails
    clusters_final = np.vectorize(final_mapping_dict.get)(clusters_intermediate, -1)
    # No warning about -1 labels

    final_cluster_labels = sorted(list(np.unique(clusters_final)))
    n_clusters_final = len(final_cluster_labels)
    n_clusters_calc = np.max(final_cluster_labels) + 1 if n_clusters_final > 0 else 0

    cluster_means_final = calculate_cluster_means(time_series_data, clusters_final, n_clusters_calc)
    return clusters_final, cluster_means_final, n_clusters_final

# --- Plotting Functions (Cleaned) ---

def plot_silhouette_scores(distance_matrix, max_k=10, random_state=RANDOM_STATE):
    """Plots silhouette scores for KMedoids. (Cleaned)"""
    silhouette_scores = []
    k_range = range(2, max_k)

    if distance_matrix.shape[0] < 2:
        return # Or raise error

    actual_k_range = [k for k in k_range if k <= distance_matrix.shape[0]]
    if not actual_k_range:
        return # Or raise error

    for k in actual_k_range:
        kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=random_state, init='k-medoids++', max_iter=300)
        labels = kmedoids.fit_predict(distance_matrix)
        if len(set(labels)) > 1:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(np.nan)

    if not silhouette_scores:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(actual_k_range, silhouette_scores, marker="o", color="red")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal K (KMedoids with DTW)")
    plt.xticks(actual_k_range)
    plt.grid(True)
    plt.show()

def plot_kmeans_elbow(data, max_k=15, random_state=RANDOM_STATE):
    """Plots the elbow curve for KMeans. (Cleaned)"""
    sum_of_squared_distances = []
    k_range = range(1, max_k)

    if data.shape[0] < 1:
        return # Or raise error

    actual_k_range = [k for k in k_range if k <= data.shape[0]]
    if not actual_k_range:
        return # Or raise error

    for k in actual_k_range:
        # KMeans will raise error on failure
        km = KMeans(n_clusters=k, random_state=random_state, init='k-means++', n_init=10)
        km = km.fit(data)
        sum_of_squared_distances.append(km.inertia_)

    if not sum_of_squared_distances:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(actual_k_range, sum_of_squared_distances, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.title('Elbow Method For Optimal k (KMeans)')
    plt.xticks(actual_k_range)
    plt.grid(True)
    plt.show()

def plot_cluster_trends_with_sinusoidal_fit(cluster_means, cluster_names, cluster_colors, savgol_window=SAVGOL_WINDOW, savgol_order=SAVGOL_ORDER):
    """Plots individual cluster trends with fits. (Cleaned)"""
    if cluster_means is None or cluster_means.ndim != 2 or cluster_means.shape[0] == 0:
        return # Or raise error

    n_clusters, n_timesteps = cluster_means.shape
    months_axis = np.arange(1, n_timesteps + 1)

    for c in range(n_clusters):
        avg_trend = cluster_means[c]
        smooth_trend = avg_trend
        valid_savgol = False
        if savgol_window < len(avg_trend) and savgol_window % 2 != 0:
            # savgol_filter will raise ValueError on failure
            smooth_trend = savgol_filter(avg_trend, window_length=savgol_window, polyorder=savgol_order)
            valid_savgol = True
        # No warnings for invalid savgol params

        params = fit_sinusoidal(months_axis, smooth_trend)
        fitted_curve = sinusoidal(months_axis, *params)

        plt.figure(figsize=(10, 5))
        plot_color = cluster_colors.get(c, 'gray')
        cluster_name = cluster_names.get(c, f'Cluster {c}')

        plt.plot(months_axis, avg_trend, linestyle="-", color=plot_color, linewidth=2, alpha=0.7, label="Raw Trend")
        if valid_savgol:
            plt.plot(months_axis, smooth_trend, linestyle="--", color="gray", linewidth=1.5, label="Smoothed Trend")
        plt.plot(months_axis, fitted_curve, linestyle="-", color="black", linewidth=1.5, alpha=0.8, label="Fitted Sinusoid")

        plt.title(f"Cosine Similarity Trend: {cluster_name} (Sinusoidal Fit)", fontsize=14)
        plt.xlabel("Time Step (e.g., Month)")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

def plot_combined_cluster_trends(cluster_means, cluster_colors, labels_dict, desired_order, save_path_svg=None, savgol_window=SAVGOL_WINDOW, savgol_order=SAVGOL_ORDER):
    """Plots all cluster trends together. (Cleaned)"""
    if cluster_means is None or cluster_means.ndim != 2 or cluster_means.shape[0] == 0:
        return # Or raise error

    n_clusters, n_timesteps = cluster_means.shape
    fig, ax = plt.subplots(figsize=(12, 7))
    months_axis = np.arange(1, n_timesteps + 1)
    plot_handles = {}

    for c in range(n_clusters):
        avg_trend = cluster_means[c]
        cluster_label_key = f"cluster {c+1}"
        plot_label = labels_dict.get(cluster_label_key, f"Cluster {c}")
        plot_color = cluster_colors.get(c, 'gray')

        smooth_trend = avg_trend
        if savgol_window < len(avg_trend) and savgol_window % 2 != 0:
            # savgol_filter will raise ValueError on failure
            smooth_trend = savgol_filter(avg_trend, window_length=savgol_window, polyorder=savgol_order)
        # No warnings

        params = fit_sinusoidal(months_axis, smooth_trend)
        fitted_curve = sinusoidal(months_axis, *params)

        line, = ax.plot(months_axis, avg_trend, linestyle="-", color=plot_color, linewidth=2.5, alpha=0.6, label=plot_label)
        ax.plot(months_axis, fitted_curve, linestyle="-", color="black", linewidth=1.2, alpha=0.9)
        plot_handles[plot_label] = line

    if n_timesteps == 47:
        year_ranges = [(0.5, 11.5), (11.5, 23.5), (23.5, 35.5), (35.5, 47.5)]
        colors = ['white', '#f0f0f0', 'white', '#f0f0f0']
        for i, (start, end) in enumerate(year_ranges):
            ax.axvspan(start, end, color=colors[i], alpha=0.2, zorder=-1)

    ax.set_xlabel("Time Step (e.g., Month)", fontsize=15)
    ax.set_ylabel("Cosine Similarity", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    # Legend creation might raise KeyError if labels mismatch, handled by default legend
    try:
        ordered_handles = [plot_handles[label] for label in desired_order if label in plot_handles]
        ordered_labels = [label for label in desired_order if label in plot_handles]
        current_labels = list(plot_handles.keys())
        for label in current_labels:
            if label not in ordered_labels:
                ordered_labels.append(label)
                ordered_handles.append(plot_handles[label])
        ax.legend(ordered_handles, ordered_labels, loc='best', fontsize=12)
    except Exception: # Catch any error during legend creation
        ax.legend(fontsize=12) # Fallback to default legend

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path_svg:
        # savefig will raise error on failure
        plt.savefig(save_path_svg, format="svg", dpi=300, bbox_inches='tight')

    plt.show()

def plot_cluster_map(shapefile_path, city_to_cluster, cluster_colors, cluster_id_col='cluster_id', city_num_col='city_number', save_path_jpg=None):
    """Plots a map colored by cluster assignment. (Cleaned)"""
    # gpd.read_file will raise error if file not found
    mg_map = gpd.read_file(shapefile_path)

    if city_num_col not in mg_map.columns:
        if city_num_col == 'city_number' and mg_map.index.name != city_num_col:
            mg_map[city_num_col] = mg_map.index
        else:
            raise KeyError(f"City identifier column '{city_num_col}' not found in shapefile.")

    # Mapping might raise TypeError, attempt conversion
    try:
        mg_map[cluster_id_col] = mg_map[city_num_col].map(city_to_cluster)
    except TypeError:
        # Conversion might raise Exception
        key_type = type(next(iter(city_to_cluster)))
        mg_map[cluster_id_col] = mg_map[city_num_col].astype(key_type).map(city_to_cluster)

    unmapped_count = mg_map[cluster_id_col].isna().sum()
    if unmapped_count > 0:
        mg_map[cluster_id_col] = mg_map[cluster_id_col].fillna(-1)
    # No warning

    # astype(int) might raise ValueError
    mg_map[cluster_id_col] = mg_map[cluster_id_col].astype(int)

    extended_colors = cluster_colors.copy()
    if -1 not in extended_colors:
        extended_colors[-1] = '#d3d3d3'
    mg_map["color"] = mg_map[cluster_id_col].map(extended_colors).fillna(extended_colors[-1])

    fig, ax = plt.subplots(figsize=(10, 8))
    mg_map.plot(ax=ax, color=mg_map["color"], edgecolor="black", linewidth=0.5, alpha=0.9)

    present_clusters = sorted([c for c in mg_map[cluster_id_col].unique() if c != -1])
    handles = []
    labels = []
    for cluster_label in present_clusters:
        color = cluster_colors.get(cluster_label)
        if color:
            handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10))
            labels.append(f'Cluster {cluster_label}')

    if handles:
        ax.legend(handles=handles, labels=labels, title="Clusters", loc='best')

    ax.set_axis_off()
    plt.title("City Clusters", fontsize=16)
    plt.tight_layout()

    if save_path_jpg:
        # savefig will raise error on failure
        plt.savefig(save_path_jpg, dpi=300, bbox_inches='tight')

    plt.show()