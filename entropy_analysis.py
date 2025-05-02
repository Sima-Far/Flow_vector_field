# -*- coding: utf-8 -*-
"""
Functions for calculating directional entropy for city/region flow data.
(Cleaned version: No try/except or prints)
"""

import numpy as np
import math

# --- Constants ---
MW = 12  # Default months window (months in a year)
N_Z_ENTROPY = 4 # Default number of discretize groups for entropy

# --- Core Logic Functions ---

def calculate_entropy(vectors_flow_map, years, num_regions, mw=MW, n_z=N_Z_ENTROPY):
    """
    Calculates directional entropy for regions over specified years.

    Args:
        vectors_flow_map (dict): Dict keyed by year, containing region vectors.
                                 Expected structure: {year: {region_id: np.array((2, mw))}}
                                 or {year: np.array((num_regions, 2, mw))} after stacking.
        years (list): List of years (strings or ints) to process.
        num_regions (int): Number of regions (e.g., 853 for cities, 66 for microregions).
        mw (int): Number of months/time steps per year.
        n_z (int): Normalization factor for entropy (number of direction types).

    Returns:
        dict: Dictionary keyed by year, containing entropy array for each region.
              {year: np.array(num_regions)}
    """
    direction_set_dic = {}
    Ent_dic = {}

    for ye in years:
        if isinstance(vectors_flow_map.get(ye), dict):
            vector_region_mean_months = np.stack(list(vectors_flow_map[ye].values()), axis=0)
        elif isinstance(vectors_flow_map.get(ye), np.ndarray):
            vector_region_mean_months = vectors_flow_map[ye]
        else:
            # If data is not dict or ndarray, this will likely raise an error later,
            # or the shape check will fail.
            continue # Or raise an error, depending on desired strictness

        if vector_region_mean_months.shape != (num_regions, 2, mw):
            # Shape mismatch, skip or raise error
            continue # Or raise ValueError

        region_direction_set = np.full((num_regions, mw), 0, dtype=int)

        for i in range(num_regions):
            for j in range(mw):
                dir_type = 0
                x = vector_region_mean_months[i, 0, j]
                y = vector_region_mean_months[i, 1, j]

                if x > 0 and y >= 0: dir_type = 1
                elif x <= 0 and y > 0: dir_type = 2
                elif x < 0 and y <= 0: dir_type = 3
                elif x >= 0 and y < 0: dir_type = 4

                region_direction_set[i, j] = dir_type

        direction_set_dic[ye] = region_direction_set

        Ent = np.full(num_regions, 0.0, dtype=float)
        for i in range(num_regions):
            plogp = 0
            counts = np.bincount(region_direction_set[i], minlength=n_z + 1)

            for direction_type in range(1, n_z + 1):
                c_e = counts[direction_type]
                if c_e > 0:
                    p = c_e / mw
                    plogp += p * math.log(p)

            if n_z > 0 and plogp != 0:
                Ent[i] = (-1.0 / n_z) * plogp
            else:
                Ent[i] = 0.0

        Ent_dic[ye] = Ent

    return Ent_dic


def add_ent(row, ent_arr, city_col='city_num'):
    """
    Helper function to add entropy value to a GeoDataFrame row based on a city identifier column.
    (Cleaned version: No try/except or prints)

    Args:
        row (pd.Series): A row from a GeoDataFrame or DataFrame.
        ent_arr (np.array): The array containing entropy values, indexed corresponding to city numbers.
        city_col (str): The name of the column in the row that contains the city identifier/index.

    Returns:
        float: The entropy value for the city in the row. Returns NaN if city number is invalid.
    """
    cityn = int(row[city_col]) # May raise ValueError/TypeError if conversion fails
    if 0 <= cityn < len(ent_arr):
        return ent_arr[cityn]
    else:
        return np.nan # Return NaN for out-of-bounds index