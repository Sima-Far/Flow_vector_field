# Flow_vector_field
This file explains the various code files in this repository and provides a guide for performing the analysis as described in the paper. Analysis are done using cities as divisions and spoke analysis repeated using micro-regions as divisions (see Main_analysis_using_functions headings Analysis using cities as initial divisions and Analysis using microregions as initial divisions .

First Step: Vector Generation from Origin-Destination (Adjacency) Matrices

The initial task is to generate vectors from OD matrices for selected spatial divisions (such as cities, municipalities, counties, etc) , including divisions with no outgoing edges (represented by zero vectors). 

Generate Vectors from the Origin-Destination Matrix: See the Python file named Generate_vector_fields.ipynb.


To generate the vector field, the following inputs are necessary:

Origin-Destination matrix (as a NumPy array). The samples are in folder named OD_matrices for two different divisions: cities and microregions.
OD/adjacency matrix is a n by n array where the ij'th entry is the weight of the vector from city i to city j
Example of matrix for 4 divisions (cities, municipalities, micro-regions, etc):
OD_matrix = np.array([[0,2,1,1],
                       [0,0,1,0],
                       [2,1,0,0],
                       [1,0,3,0]])

Latitude and longitude of divisions from the OD matrix (as a NumPy array) which we named coordiantes:
Example of coordinates for 4 divisions (cities, municipalities, micro-regions, etc) :
coordinates = np.array([[0,0], [1,0], [0,1], [1,1]])
coordinates is a n by 2 array where the i'th entry is the coordinates of the i'th city [x, y]

Shapefile containing the map boundaries of divisions. The samples are in folder named for two different divisions: cities and microregions.



A function in the Python file named generate_vector_for_each_location calculates vectors for each location (origin), assuming the mean or sum of outgoing edges. This function also improves vector accuracy by adding 100 points from the boundary to reduce boundary effects and avoid vectors pointing outside the map due to the boundary, ensuring that vectors are calculated based on actual edge/trade behavior.

Vector Field Visualization: See the Python file named Main_analysis_using_functions.ipynb Heading “Visualise vector fields on map - Flow map visualisation” .

Second Step: Analysis on Resulting Vector Fields

Once the vector field is generated, several analysis steps can be performed on it.
Cosine Similarity Calculation: See the Python file named Cosine_similarity_function.ipynb for functions and heading calculate cosine similarity  in Main_analysis_using_functions.ipynb 

This function calculates the cosine similarity between consecutive time frames (e.g., 12 adjacency matrices for 12 months, weeks, or days). It then applies k-means clustering to group locations based on their cosine similarity behavior.


Entropy Calculation: See the Python file named entropy_calculation_function.ipynb for functions and heading Entropy calculation in Main_analysis_using_functions.ipynb.


Spatial Autocorrelation Analysis and Spatial Lag and Moran’s I Calculation: See the Python file named [filename] for functions and heading Spatial autocorrelation, Moran's I (global), and spatial lag in Main_analysis_using_functions.ipynb.

Interpolating Vectors for Different Scales: See the Python file named [filename] for functions and in Main_analysis_using_functions.ipynb.

This function uses the previous inputs—origin-destination matrix, location lat/lon, and the shapefile—along with an integer value n to generate an n x n grid on the map. It calculates vectors for all cells in this grid, which can be either smaller or larger than the initial divisions, depending on the scale.

The process also covers how to interpolate a vector field across different selected scales based on number of cells.

