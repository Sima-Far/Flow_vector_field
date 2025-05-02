# -*- coding: utf-8 -*-
"""
Main script to orchestrate the analysis workflow for city flow data,
importing functions from specialized modules.
"""

# --- Imports ---
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt # Keep for potential direct plotting if needed

# Import functions from the refactored modules
import entropy_analysis
import similarity_clustering
import spatial_autocorrelation

# --- Constants and Configuration ---
# Define constants needed for the main workflow orchestration
# Or load them from a configuration file

# File Paths (MAKE SURE THESE ARE CORRECT)
SHAPEFILE_PATH = r'C:/sima/PhD/Thesis/Mg_dataset/31MUE250GC_SIR.shp' # Example path
VECTOR_FILE_TEMPLATE = r'flow_map/vector_files_reshape/vector_city_sum_month_{}.txt' # Example path
# COSINE_SIM_DATA_PATH = 'path/to/your/cosine_similarity_data.txt' # Path to pre-calculated cosine similarity
# ENTROPY_VECTOR_DATA_PATH = 'path/to/your/entropy_vector_data.pkl' # Or however entropy vectors are stored

# Analysis Parameters
YEARS_LIST = ['2013', '2014', '2015', '2016'] # Example years
NUM_REGIONS = 853 # Example: Number of cities/regions
TOTAL_MONTHS_SIMILARITY = 47 # Example: Number of time steps in cosine similarity data
MONTHS_PER_YEAR = 12 # Standard months per year

# Clustering Parameters (can be imported from similarity_clustering or defined here)
N_CLUSTERS_INITIAL = similarity_clustering.DEFAULT_N_CLUSTERS_INITIAL
RANDOM_STATE = similarity_clustering.RANDOM_STATE
# Remapping rules (specific to the analysis)
CLUSTER_REMAP_RULES = [(1, 3)] # Rule: clusters[clusters == 1] = 3
CLUSTER_FINAL_MAPPING_DICT = {0: 3, 2: 2, 3: 1, 4: 0} # Maps intermediate labels {0,2,3,4} to final {0,1,2,3}

# Spatial Analysis Parameters
SPATIAL_WEIGHTS_METHOD = 'queen' # 'queen', 'rook', or 'knn'
SPATIAL_WEIGHTS_K = None # Specify k if method is 'knn'
MORAN_PERMUTATIONS = 999 # Number of permutations for Moran's I significance

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate the analysis workflow."""
    print("Starting analysis workflow...")

    # --- !! IMPORTANT: Load Your Data Here !! ---
    # Load data required for the different stages.
    # This might involve loading pre-calculated results or raw data.

    # Example: Load cosine similarity data (replace with actual loading)
    try:
        # This data needs to exist or be calculated beforehand if not done in this script
        # vector_mean_city_cs = np.loadtxt(COSINE_SIM_DATA_PATH)
        # Placeholder: Generate random data if file doesn't exist
        print(f"Placeholder: Generating random cosine similarity data ({NUM_REGIONS}x{TOTAL_MONTHS_SIMILARITY})")
        vector_mean_city_cs = np.random.rand(NUM_REGIONS, TOTAL_MONTHS_SIMILARITY)
        n_cities_loaded, n_months_loaded = vector_mean_city_cs.shape
        if n_cities_loaded != NUM_REGIONS or n_months_loaded != TOTAL_MONTHS_SIMILARITY:
            print(f"Warning: Loaded/Generated cosine data shape ({n_cities_loaded}x{n_months_loaded}) doesn't match expected ({NUM_REGIONS}x{TOTAL_MONTHS_SIMILARITY}).")
            # Adjust NUM_REGIONS/TOTAL_MONTHS_SIMILARITY or handle error
            # For placeholder, we'll proceed with the generated shape
            # NUM_REGIONS = n_cities_loaded
            # TOTAL_MONTHS_SIMILARITY = n_months_loaded

        print(f"Using cosine similarity data: {NUM_REGIONS} regions, {TOTAL_MONTHS_SIMILARITY} time steps.")

    except FileNotFoundError:
        print(f"Error: Cosine similarity data file not found at {COSINE_SIM_DATA_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during cosine similarity data loading: {e}. Exiting.")
        return

    # Example: Load vectors needed for entropy (replace with actual loading)
    # This could be a dictionary loaded from a pickle file or constructed
    # vectors_flow_map_data = load_entropy_vectors(ENTROPY_VECTOR_DATA_PATH) # Replace with your loading function
    # Placeholder: Generate random data
    print(f"Placeholder: Generating random vector data for entropy ({len(YEARS_LIST)} years)")
    vectors_flow_map_data = {
        year: np.random.rand(NUM_REGIONS, 2, MONTHS_PER_YEAR) * 10 - 5 # Random vectors centered around 0
        for year in YEARS_LIST
    }
    # --- End Data Loading Section ---


    # --- Stage 1: Entropy Calculation (Optional) ---
    run_entropy = True # Set to False to skip
    if run_entropy:
        print("\n--- Stage 1: Entropy Calculation ---")
        ent_dic_cities = entropy_analysis.calculate_entropy(
            vectors_flow_map_data,
            YEARS_LIST,
            num_regions=NUM_REGIONS,
            mw=MONTHS_PER_YEAR
        )
        # Example: Print some results
        if ent_dic_cities and YEARS_LIST[0] in ent_dic_cities:
            print(f"Entropy for Regions (Example Year {YEARS_LIST[0]}, First 5):", ent_dic_cities[YEARS_LIST[0]][:5])
        else:
            print("Entropy calculation did not produce results.")
        # You might want to add this entropy data to a GeoDataFrame later
        # Example (requires loading gdf first):
        # gdf = gpd.read_file(SHAPEFILE_PATH)
        # gdf['entropy_2013'] = gdf.apply(entropy_analysis.add_ent, axis=1, ent_arr=ent_dic_cities.get('2013', []), city_col='city_number') # Adjust city_col if needed


    # --- Stage 2: Clustering based on Cosine Similarity ---
    run_clustering = True # Set to False to skip
    if run_clustering:
        print("\n--- Stage 2: Clustering ---")
        # Calculate DTW distance matrix from cosine similarity time series
        distance_matrix = similarity_clustering.calculate_dtw_distance_matrix(vector_mean_city_cs)

        if distance_matrix is not None:
            # Optional: Analyze optimal k
            # similarity_clustering.plot_silhouette_scores(distance_matrix, max_k=12, random_state=RANDOM_STATE)
            # similarity_clustering.plot_kmeans_elbow(vector_mean_city_cs, max_k=15, random_state=RANDOM_STATE) # Note: KMeans uses Euclidean

            # Perform KMedoids clustering
            clusters_initial, kmedoids_model = similarity_clustering.perform_kmedoids_clustering(
                distance_matrix,
                n_clusters=N_CLUSTERS_INITIAL,
                random_state=RANDOM_STATE
            )

            if clusters_initial is not None:
                city_to_cluster_initial = {i: clusters_initial[i] for i in range(NUM_REGIONS)}

                # Calculate initial cluster means
                cluster_means_initial = similarity_clustering.calculate_cluster_means(
                    vector_mean_city_cs,
                    clusters_initial,
                    N_CLUSTERS_INITIAL
                )

                # Optional: Plot initial cluster trends
                # similarity_clustering.plot_cluster_trends_with_sinusoidal_fit(
                #     cluster_means_initial,
                #     similarity_clustering.CLUSTER_NAMES_INITIAL, # Using names from module
                #     similarity_clustering.CLUSTER_COLORS_INITIAL # Using colors from module
                # )

                # --- Stage 3: Remap Clusters and Plot Final Results ---
                print("\n--- Stage 3: Remapping and Final Plots ---")

                clusters_final, cluster_means_final, n_clusters_final = similarity_clustering.remap_clusters_and_recalculate_means(
                    clusters_initial,
                    vector_mean_city_cs,
                    CLUSTER_REMAP_RULES,
                    CLUSTER_FINAL_MAPPING_DICT
                )

                if clusters_final is not None and cluster_means_final is not None:
                    city_to_cluster_final = {i: clusters_final[i] for i in range(NUM_REGIONS)}

                    # Plot combined final cluster trends
                    similarity_clustering.plot_combined_cluster_trends(
                        cluster_means_final,
                        similarity_clustering.REMAPPED_CLUSTER_COLORS, # Using remapped colors from module
                        similarity_clustering.REMAPPED_LABELS,         # Using remapped labels from module
                        similarity_clustering.DESIRED_LEGEND_ORDER,    # Using legend order from module
                        save_path_svg='clusters_curves_final.svg'      # Output path
                    )

                    # Plot final cluster map
                    # Ensure city_num_col matches the identifier used in city_to_cluster_final keys (likely the index 0..N-1)
                    similarity_clustering.plot_cluster_map(
                        SHAPEFILE_PATH,
                        city_to_cluster_final,
                        similarity_clustering.REMAPPED_CLUSTER_COLORS, # Use remapped colors
                        city_num_col='city_number', # Adjust if shapefile uses a different ID column mapped 0..N-1
                        save_path_jpg='clusters_map_final.jpg' # Output path
                    )
                else:
                    print("Cluster remapping failed. Skipping final plots.")
            else:
                print("Initial clustering failed. Skipping remapping and final plots.")
        else:
            print("DTW distance matrix calculation failed. Skipping clustering.")


    # --- Stage 4: Moran's I Analysis ---
    run_morans = True # Set to False to skip
    if run_morans:
        print("\n--- Stage 4: Moran's I Analysis ---")
        # Calculate monthly Moran's I for all years specified
        moran_dict_results = spatial_autocorrelation.calculate_monthly_moran_for_years(
            YEARS_LIST,
            VECTOR_FILE_TEMPLATE,
            SHAPEFILE_PATH,
            NUM_REGIONS,
            mw=MONTHS_PER_YEAR,
            weights_method=SPATIAL_WEIGHTS_METHOD,
            k_knn=SPATIAL_WEIGHTS_K,
            moran_kwargs={'permutations': MORAN_PERMUTATIONS} # Pass permutation setting
        )

        # Plot Moran's I heatmap
        if moran_dict_results: # Check if results were generated
            spatial_autocorrelation.plot_moran_heatmap(
                moran_dict_results,
                YEARS_LIST,
                list(range(MONTHS_PER_YEAR)), # Months 0-11
                save_path='moran_I_heatmap.pdf' # Output path
            )
        else:
            print("Monthly Moran's I calculation failed or produced no results. Skipping heatmap.")

        # Optional: Detailed Moran's I analysis for a specific month/year (e.g., Jan 2013)
        run_detailed_morans = False # Set to True to run detailed example
        if run_detailed_morans and YEARS_LIST:
            target_year = YEARS_LIST[0]
            target_month_idx = 0 # January
            print(f"\nDetailed Moran's I for {target_year}-{target_month_idx+1}:")
            try:
                # Load data for the specific year/month
                vec_year = np.loadtxt(VECTOR_FILE_TEMPLATE.format(target_year)).reshape((NUM_REGIONS, 2, MONTHS_PER_YEAR))
                gdf_detailed = gpd.read_file(SHAPEFILE_PATH)
                # Add consistent ID if needed
                if 'region_id' not in gdf_detailed.columns:
                    gdf_detailed['region_id'] = gdf_detailed.index
                id_col = 'region_id'

                # Calculate vector size for the target month
                gdf_detailed['vector_size'] = np.linalg.norm(vec_year[:, :, target_month_idx], axis=1)

                # Calculate weights (can reuse if already calculated and stored)
                wq_detailed = spatial_autocorrelation.calculate_spatial_weights(
                    gdf_detailed, method=SPATIAL_WEIGHTS_METHOD, k=SPATIAL_WEIGHTS_K, id_variable=id_col
                )

                if wq_detailed:
                    # Calculate Moran's I
                    moran_detailed = spatial_autocorrelation.calculate_moran_I(
                        gdf_detailed, 'vector_size', wq_detailed, permutations=MORAN_PERMUTATIONS
                    )
                    # Calculate Spatial Lag
                    gdf_detailed = spatial_autocorrelation.calculate_spatial_lag(
                        gdf_detailed, 'vector_size', wq_detailed
                    )

                    # Plot results for the detailed analysis
                    spatial_autocorrelation.plot_geodataframe_heatmap(
                        gdf_detailed, 'vector_size', f'Vector Size Heatmap ({target_year}-{target_month_idx+1})',
                        save_path=f'vector_size_heatmap_{target_year}_{target_month_idx+1}.pdf'
                    )
                    spatial_autocorrelation.plot_spatial_lag_heatmap(
                        gdf_detailed, 'spatial_lag', f'Spatial Lag Heatmap ({target_year}-{target_month_idx+1})',
                        save_path=f'spatial_lag_heatmap_{target_year}_{target_month_idx+1}.pdf'
                    )
                    if moran_detailed:
                        spatial_autocorrelation.plot_moran_scatterplot(
                            moran_detailed, variable_col='vector_size', # Pass var name for labels
                            save_path=f'moran_scatterplot_{target_year}_{target_month_idx+1}.pdf'
                        )
                        if moran_detailed.permutations:
                            spatial_autocorrelation.plot_moran_simulation(
                                moran_detailed,
                                save_path=f'moran_simulation_{target_year}_{target_month_idx+1}.pdf'
                            )
                else:
                    print("Failed to calculate weights for detailed analysis.")

            except FileNotFoundError:
                print(f"Error: Data file not found for detailed analysis of year {target_year}.")
            except Exception as e:
                print(f"Error during detailed Moran's I analysis for {target_year}-{target_month_idx+1}: {e}")


    print("\nAnalysis workflow finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Note: Ensure all file paths and parameters at the top are correct.
    # Implement actual data loading in the designated section within main().
    main()
    # print("Script loaded. Implement data loading in main() and adjust parameters before running.")