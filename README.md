# Flow_vector_field
This file explains the various code files in this repository and provides a guide for performing the analysis as described in the paper.
First Step: Vector Generation from Adjacency Matrices
The initial task is to generate vectors from adjacency matrices for selected spatial divisions, including divisions with no outgoing edges (represented by zero vectors). The process also covers how to interpolate a vector field across different selected scales.
Generate Vectors from the Origin-Destination Matrix: See the Python file named [filename].


To generate the vector field, the following inputs are necessary:
Origin-Destination matrix (as a NumPy array)


Latitude and longitude of locations from the adjacency matrix (as a NumPy array)


Shapefile containing the map boundary


A function in the Python file named generate_vector_for_each_location calculates vectors for each location (origin), assuming the mean or sum of outgoing edges. This function also improves vector accuracy by adding 100 points from the boundary to reduce boundary effects and avoid vectors pointing outside the map due to the boundary, ensuring that vectors are calculated based on actual edge/trade behavior.
Interpolating Vectors for Different Scales: See the Python file named [filename].


This function uses the previous inputs—origin-destination matrix, location lat/lon, and the shapefile—along with an integer value n to generate an n x n grid on the map. It calculates vectors for all cells in this grid, which can be either smaller or larger than the initial divisions, depending on the scale.
Vector Field Visualization: See the Python file named [filename].


Second Step: Analysis on Resulting Vector Fields
Once the vector field is generated, several analysis steps can be performed on it.
Cosine Similarity Calculation: This function calculates the cosine similarity between consecutive time frames (e.g., 12 adjacency matrices for 12 months, weeks, or days). It then applies k-means clustering to group locations based on their cosine similarity behavior.


Entropy Calculation: See the Python file named [filename].


Spatial Autocorrelation Analysis: See the Python file named [filename].


Spatial Lag and Moran’s I Calculation: See the Python file named [filename].

