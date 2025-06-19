# Flow_vector_field

This file provides an overview of the code files in this repository and serves as a guide to performing the analysis described in the paper. The analysis is conducted using cities as the initial spatial divisions, and then repeated using micro-regions (see the Main_analysis_using_functions.ipynb under the headings: "Analysis using cities as initial divisions" and "Analysis using micro-regions as initial divisions").
________________________________________
Step 1: Vector Generation from Origin-Destination (Adjacency) Matrices
The first step involves generating vectors from Origin-Destination (OD) matrices for selected spatial divisions (e.g., cities, municipalities, or counties), including divisions with no outgoing edges (represented as zero vectors).

Script: See the Python notebook Generate_vector_fields.ipynb. The generated vector fields are in generated_vector_fields folder.

Inputs required:
1.	Origin-Destination Matrix
A NumPy array where the entry at row i, column j represents the weight of movement from division i to division j.
Example:
OD_matrix = np.array([[0, 2, 1, 1],
                      [0, 0, 1, 0],
                      [2, 1, 0, 0],
                      [1, 0, 3, 0]])

Sample matrices for cities and micro-regions are provided in the OD_matrices folder.
Coordinates of Divisions

2. A NumPy array of latitude and longitude values for each division. Samples for cities and micro-regions are provided in the Coordinates folder.

Example:
coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

Here, each entry [x, y] corresponds to the coordinates of one division.

3. Shapefile of Divisions
Shapefiles containing the geographic boundaries for each division (cities and micro-regions) are also provided (in polygon_shapefiles folder).

A function in generate_vector_for_each_location.py computes vectors for each origin location, using either the mean or sum of outgoing edges. To improve boundary accuracy, the function adds 100 boundary points, minimizing edge effects and preventing vectors from pointing outside the map.

Vector Field Visualization: See Main_analysis_using_functions.ipynb under the heading “Visualise vector fields on map - Flow map visualisation”.
________________________________________

Step 2: Analysis on Resulting Vector Fields
After generating the vector field, several types of analyses can be conducted:

Cosine Similarity
Computes cosine similarity between vectors across consecutive time frames (e.g., monthly or weekly matrices). This helps identify spatial patterns over time. K-means clustering is then applied to group locations with similar temporal trends.
      See: Cosine_similarity_function.ipynb and the corresponding section in Main_analysis_using_functions.ipynb.

Entropy Calculation
Measures the randomness or disorder of movement patterns.
   See: entropy_calculation_function.ipynb and relevant section in Main_analysis_using_functions.ipynb.


Spatial Autocorrelation & Moran’s I
Analyzes spatial dependency and clustering of vector directions and magnitudes.
      See: the relevant script and the “Spatial autocorrelation, Moran’s I (global), and spatial lag” section in Main_analysis_using_functions.ipynb.
