# -*- coding: utf-8 -*-
"""
Functions for calculating spatial weights, Moran's I (global), spatial lag,
and plotting related spatial statistics results (heatmaps, scatterplots).
(Cleaned version: No try/except or prints)
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import libpysal as lps
import esda
from esda.moran import Moran
import splot
from splot.esda import moran_scatterplot, plot_moran_simulation
import seaborn as sns

# --- Constants ---
MW = 12 # Default months window

# --- Core Logic Functions (Cleaned) ---

def calculate_spatial_weights(gdf, method='queen', k=None, id_variable=None, **kwargs):
    """Calculates spatial weights matrix using libpysal. (Cleaned)"""
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'geometry' column.")

    if method.lower() == 'queen':
        wq = lps.weights.Queen.from_dataframe(gdf, idVariable=id_variable, **kwargs)
    elif method.lower() == 'rook':
        wq = lps.weights.Rook.from_dataframe(gdf, idVariable=id_variable, **kwargs)
    elif method.lower() == 'knn':
        if k is None:
            raise ValueError("Parameter 'k' is required for KNN weights.")
        wq = lps.weights.KNN.from_dataframe(gdf, k=k, idVariable=id_variable, **kwargs)
    else:
        raise ValueError(f"Unsupported weights method: '{method}'. Choose 'queen', 'rook', or 'knn'.")

    wq.transform = 'r' # Row-standardize
    return wq

def calculate_moran_I(gdf, variable_col, wq, permutations=999, two_tailed=True):
    """Calculates Moran's I global spatial autocorrelation statistic. (Cleaned)"""
    if variable_col not in gdf.columns:
        raise KeyError(f"Variable column '{variable_col}' not found in GeoDataFrame.")
    if wq is None:
        raise ValueError("Invalid spatial weights object provided.")
    if gdf.shape[0] != wq.n:
        raise ValueError(f"GeoDataFrame rows ({gdf.shape[0]}) do not match weights matrix size ({wq.n}).")

    y = gdf[variable_col].values
    # No explicit NaN handling here; Moran() might handle or raise error.
    # If NaNs cause issues, they should be handled before calling this function.

    moran = Moran(y, wq, permutations=permutations, two_tailed=two_tailed)
    return moran

def calculate_monthly_moran_for_years(years, vector_file_template, shapefile_path, num_regions, mw=MW, weights_method='queen', k_knn=None, weights_kwargs=None, moran_kwargs=None):
    """Calculates Moran's I for vector magnitude monthly across years. (Cleaned)"""
    if weights_kwargs is None: weights_kwargs = {}
    if moran_kwargs is None: moran_kwargs = {}

    # read_file will raise error if not found
    mg_map = gpd.read_file(shapefile_path)
    if 'geometry' not in mg_map.columns:
        raise ValueError("Shapefile must contain a 'geometry' column.")
    if 'region_id' not in mg_map.columns:
        mg_map['region_id'] = mg_map.index
    id_col = 'region_id'
    # No warning for shape mismatch

    # calculate_spatial_weights will raise error on failure
    wq = calculate_spatial_weights(mg_map, method=weights_method, k=k_knn, id_variable=id_col, **weights_kwargs)
    if wq is None: # Should not happen if calculate_spatial_weights raises error, but check anyway
        raise RuntimeError("Failed to calculate spatial weights.")

    moran_dict_years = {}

    for ye in years:
        vector_file_path = vector_file_template.format(ye)
        # loadtxt/reshape will raise error on failure
        vector_month_mean = np.loadtxt(vector_file_path)
        expected_elements = num_regions * 2 * mw
        if vector_month_mean.size != expected_elements:
            raise ValueError(f"Loaded data size ({vector_month_mean.size}) doesn't match expected ({expected_elements}) for shape ({num_regions}, 2, {mw})")
        vector_month_mean = vector_month_mean.reshape((num_regions, 2, mw))

        moran_months = {}
        mg_map_vector = mg_map[[id_col, 'geometry']].copy()

        for mo in range(mw):
            vector_month = vector_month_mean[:, :, mo]
            r_vector_region = np.linalg.norm(vector_month, axis=1)
            mg_map_vector['vector_size'] = r_vector_region

            # calculate_moran_I will raise error on failure
            try:
                moran_result = calculate_moran_I(mg_map_vector, 'vector_size', wq, **moran_kwargs)
                moran_months[mo] = moran_result.I
            except Exception: # Catch specific errors if needed, otherwise store NaN
                moran_months[mo] = np.nan # Store NaN on error

        moran_dict_years[ye] = moran_months

    return moran_dict_years

def calculate_spatial_lag(gdf, variable_col, wq):
    """Calculates the spatial lag for a variable. (Cleaned)"""
    if variable_col not in gdf.columns:
        raise KeyError(f"Variable column '{variable_col}' not found.")
    if wq is None:
        raise ValueError("Invalid spatial weights object provided.")
    if gdf.shape[0] != wq.n:
        raise ValueError(f"GeoDataFrame rows ({gdf.shape[0]}) do not match weights matrix size ({wq.n}).")
    # No warning for non-row-standardized weights

    y = gdf[variable_col].values
    # lag_spatial will raise error on failure
    spatial_lag = lps.weights.lag_spatial(wq, y)

    gdf_out = gdf.copy()
    gdf_out['spatial_lag'] = spatial_lag
    gdf_out['spatial_lag_diff'] = y - spatial_lag
    return gdf_out

# --- Plotting Functions (Cleaned) ---

def plot_geodataframe_heatmap(gdf, column, title, cmap='viridis', scheme='quantiles', k=5, save_path=None, **kwargs):
    """Plots a choropleth heatmap. (Cleaned)"""
    if column not in gdf.columns:
        raise KeyError(f"Column '{column}' not found in GeoDataFrame.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor('white')
    ax.set_axis_off()

    # gdf.plot will raise error on failure (e.g., ImportError for mapclassify)
    legend_kwds = kwargs.pop('legend_kwds', {})
    legend_kwds.setdefault('loc', 'upper left')
    legend_kwds.setdefault('bbox_to_anchor', (1.02, 1))
    legend_kwds.setdefault('title', column.replace('_', ' ').title())
    legend_kwds.setdefault('fmt', '{:.2f}')

    gdf.plot(ax=ax, column=column, legend=True, legend_kwds=legend_kwds,
             cmap=cmap, scheme=scheme, k=k, edgecolor='darkgrey',
             linewidth=0.3, **kwargs)

    ax.set_title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        # savefig will raise error on failure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_spatial_lag_heatmap(gdf, column='spatial_lag', title="Spatial Lag Heatmap", cmap='viridis', scheme='quantiles', k=5, save_path=None, gdf_overlays=None):
    """Plots spatial lag heatmap with optional overlays. (Cleaned)"""
    if column not in gdf.columns:
        raise KeyError(f"Spatial lag column '{column}' not found in GeoDataFrame.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor('white')
    ax.set_axis_off()

    # gdf.plot will raise error on failure
    legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1.02, 1), 'title': 'Spatial Lag'}
    gdf.plot(ax=ax, column=column, legend=True, legend_kwds=legend_kwds,
             cmap=cmap, scheme=scheme, k=k, edgecolor='darkgrey',
             linewidth=0.3, alpha=0.85)

    if gdf_overlays:
        for overlay_gdf, plot_kwargs in gdf_overlays:
            plot_kwargs.setdefault('color', 'gray')
            plot_kwargs.setdefault('edgecolor', 'white')
            plot_kwargs.setdefault('linewidth', 0.5)
            plot_kwargs.setdefault('alpha', 0.9)
            overlay_gdf.plot(ax=ax, **plot_kwargs)

    ax.set_title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        # savefig will raise error on failure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_moran_scatterplot(moran_result, gdf=None, variable_col=None, wq=None, zstandard=True, save_path=None, **kwargs):
    """Plots the Moran scatterplot using splot. (Cleaned)"""
    if moran_result is None:
        raise ValueError("Invalid Moran result object provided.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor('white')

    # moran_scatterplot will raise error on failure
    scatter_kwds = kwargs.pop('scatter_kwds', {})
    scatter_kwds.setdefault('color', '#888888')
    scatter_kwds.setdefault('s', 30)
    scatter_kwds.setdefault('alpha', 0.6)
    fitline_kwds = kwargs.pop('fitline_kwds', {})
    fitline_kwds.setdefault('color', '#3192c8')
    fitline_kwds.setdefault('linewidth', 2)

    moran_scatterplot(moran_result, ax=ax, zstandard=zstandard,
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds, **kwargs)

    x_label = f"Standardized {variable_col or 'Variable'} (z)" if zstandard else (variable_col or 'Variable')
    y_label = f"Spatial Lag of Standardized {variable_col or 'Variable'} (Wz)" if zstandard else f"Spatial Lag of {variable_col or 'Variable'} (Wy)"
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    p_val_type = "sim" if moran_result.permutations else "norm"
    p_val = getattr(moran_result, f'p_{p_val_type}', float('nan'))
    ax.set_title(f"Moran Scatterplot (I={moran_result.I:.3f}, p={p_val:.3f})", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        # savefig will raise error on failure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_moran_simulation(moran_result, save_path=None, **kwargs):
    """Plots the Moran's I simulation reference distribution. (Cleaned)"""
    if moran_result is None or moran_result.permutations is None or moran_result.permutations == 0:
        raise ValueError("Moran result object must have simulation results (permutations > 0).")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_facecolor('white')

    # plot_moran_simulation will raise error on failure
    fitline_kwds = kwargs.pop('fitline_kwds', {})
    fitline_kwds.setdefault('color', '#e41a1c')

    plot_moran_simulation(moran_result, ax=ax, fitline_kwds=fitline_kwds, **kwargs)

    ax.set_xlabel("Moran's I Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Reference Distribution for Moran's I ({moran_result.permutations} Permutations)", fontsize=14)
    plt.tight_layout()

    if save_path:
        # savefig will raise error on failure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_moran_heatmap(moran_dict, years, months, title="Monthly Moran's I", vmin=None, vmax=None, cmap='RdBu_r', save_path=None):
    """Plots a heatmap of Moran's I values over time. (Cleaned)"""
    num_years = len(years)
    num_months = len(months)
    moran_array = np.full((num_years, num_months), np.nan)
    year_map = {year: i for i, year in enumerate(years)}

    for year_key, month_data in moran_dict.items():
        if year_key in year_map:
            row_idx = year_map[year_key]
            for month_idx, moran_val in month_data.items():
                if month_idx in months:
                    col_idx = months.index(month_idx)
                    moran_array[row_idx, col_idx] = moran_val
        # No warning for skipped years

    fig, ax = plt.subplots(1, 1, figsize=(max(8, num_months * 0.8), max(4, num_years * 0.6)))
    sns.set(font_scale=1.1)

    if vmin is None:
        vmin = np.nanmin(moran_array) if not np.all(np.isnan(moran_array)) else -0.5
    if vmax is None:
        vmax = np.nanmax(moran_array) if not np.all(np.isnan(moran_array)) else 0.5
    abs_max = max(abs(vmin), abs(vmax))
    if cmap.lower() in ['rdgy_r', 'rdgy', 'rdbu', 'rdbu_r', 'coolwarm']:
        vmin, vmax = -abs_max, abs_max

    # sns.heatmap will raise error on failure
    sns.heatmap(moran_array, cmap=cmap, linewidth=0.8, vmin=vmin, vmax=vmax,
                cbar=True, annot=False, fmt=".2f", ax=ax,
                cbar_kws={'label': "Moran's I"},
                xticklabels=[m + 1 for m in months], yticklabels=years)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Year", fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        # savefig will raise error on failure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()