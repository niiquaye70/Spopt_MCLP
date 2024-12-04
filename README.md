# To Begin 
### Data Prep 
**Download** the zipped file on the main branch, titled "SPOPT_DATA.ZIP" 

### Code Prep
**Download** the zipped file on the main branch, titled "SPOPT_CODE.ZIP"

# Max-P Regionalization

## Abstract Explanation
**MAX-P** is a clustering algorithm which clusters pre-defined geographic areas (in our case, census tracts) 

**Threshold** 
-The Max-P Threshold is a minumum threshold of a certain attribute that each group must meet
IE: if population is the variable, and threshold is at 500, it will regionalize so that every region has an poulation at least 500 people
-In our case, we can take advantage of this to make sure each grouping of census tracts as at least 40 tracts 

**Attribute** 
-We can select one attribute which we want to *homogenous* within each group (that is say, the lowest possible variance given the maximum/threshold constraints). 
-In our case, it will be income

**Maximum**
-The algorithm will attempt to create the maximum number of spatial units given the threshold, attribute and queen weight. This is becasaue the more spatial units we have, the more homogenity we have as well, giving us a clearer picture of our data. 

## Implementation 

### Step 1: Load Libraries 
```python
import geopandas as gpd
import libpysal
from spopt.region import MaxPHeuristic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy
````

### Step 2: Initialize random seed to randomize intial cluster assignments
```python
random_seed = 123456
````

### Step 3: Load + Clean + Project shapefile
```python
shapefile_path = "SPOPT_DATA_JOINED.shp"
gdf = gpd.read_file(shapefile_path)
gdf["spopt_v244"] = pd.to_numeric(gdf["spopt_v244"], errors="coerce")
gdf["spopt_v244"] = gdf["spopt_v244"].fillna(gdf["spopt_v244"].median())
gdf = gdf.to_crs("EPSG:2272")

### Step 4: Create a spatial weights matrix using Queen contiguity
w = libpysal.weights.Queen.from_dataframe(gdf)
```python
### Step 5: Define parameters for Max-P 
attrs_name = ["spopt_v244"] #attribute we want to be homogenous, in this case income inequality
gdf["count"] = 1 # assings a count of 1 to each census tract
threshold_name = "count" # points algorithm to count for threshold attribute
threshold = 40  # sets threshold to 40 "counts" per region (each census tract is one count, so effectively 40 tracts)
top_n = 10 # controls how many tracts are evaluated at each step. When selecting this, you are balancing computational cost and accuracy (however, only top candidates are included in the top_n)


### Step 6: Implement random seed to randomize intial cluster assignments
```python
np.random.seed(random_seed)
````

### Step 7: initialize and solve the model
```python
model = MaxPHeuristic(gdf, w, attrs_name, threshold_name, threshold, top_n)
model.solve()
````


### Step 8: Assign the region labels back to the shapefile
```python
gdf["region"] = model.labels_
````

### Step 9 (optional): Print histogram showing distribution of tracts
```python
plt.figure(figsize=(10, 6))
gdf["region"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Region Label")
plt.ylabel("Number of Census Tracts")
plt.title("Distribution of Census Tracts Across Resulting Clusters")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.savefig("cluster_histogram.png", dpi=300, bbox_inches="tight")
plt.show()
````

### Step 10: Visualize resulting regions
```python
gdf.plot(column="region", legend=True, cmap="Set3", edgecolor="black")
plt.title("Max-P Clustering of Census Tracts by Income per Capita")
plt.savefig("maxp_map.png", dpi=300, bbox_inches="tight")
plt.show()
````

### Step 11: Print a table w/ income per capita in each region
```python
cluster_summary = gdf.groupby("region").agg(
    average_income_per_capita=("spopt_v244", "mean"),
    number_of_census_tracts=("region", "count")
).reset_index()
cluster_summary.columns = ["Region", "Average Income per Capita", "Number of Census Tracts"]
print(cluster_summary)
````

### Step 12 (optional): Create a shapefile of the poorest region
```python
poorest_region_label = cluster_summary.loc[cluster_summary["Average Income per Capita"].idxmin(), "Region"]
poorest_region_gdf = gdf[gdf["region"] == poorest_region_label]
output_shapefile_path = "poorest.shp"
poorest_region_gdf.to_file(output_shapefile_path)
````
# Skater Regionalization Algortihm

## Explanation Part One - Tree Theory 

### Tree Data Structure 

* A disadvantage of arrays/linked lists is their linear nature, meaning that in order to find an item we must search through each preceding item. This is computationally and temporally inefficient.
* Instead, we can use a “tree structure”. This is a series of nodes (leaves) connected by edges (branches).
* Every node is connected to the root direction by exactly one edge, moving parent to child. A parent can have multiple children, but child can only have one parent.

### Spanning Trees
* A minimum spanning tree structure is a tree network which connects all nodes within the network, without any circuits/loops (for efficiency)
* So within the ST network, there exists a path from any other one node to any other node, but there are no loops.
* A “minimum cost spanning tree” is a spanning tree network which has been optimized such that all nodes are connected, without loops, and using the minimum number of edges possible.
* A MCST can be further optimized - sometimes we don’t need all our nodes to be connected (this is highly dependent on your application of the MCST), so we can remove edges and therefore connections. This process is called “pruning” and forms the basis for the SKATER algorithm.

## Abstract Explanation Part Two - SKATER
* The Skater Algorithm starts with a contiguous MCST
* This tree represents a continous geographic phenomena, however in our case the data will be “psuedo-continuous”, because we are working with census tract data. That is to say even if we are working with polygons, it can be thought of as VERY low resolution, unpartitioned raster data.
* When we implement SKATER, the MCST will now be “pruned”, by removing edges (connections) which link disimilar regions (census tracts)
* This is iteratively carried out, and eventually we are left with different groups of linked regions, which form the regions of the SKATER outputs.
**Important Model Parameters**
*attrs_name* - The attribute which we want to be homogenous within regions
*n_clusters* - The overall number of contiguous regions we want after pruning
*floor* - minimum number of spatial objects in each region

## Implementation 

### Step 1: Load Libaries
```python
import geopandas as gpd
import pandas as pd
import libpysal
import matplotlib.pyplot as plt
import numpy
import pandas
import shapely
from sklearn.metrics import pairwise as skm
import spopt
import warnings
````
### Step 2: Load + Clean + Project Data

```python
shapefile_path = “SPOPT_DATA_JOINED.shp”
gdf = gpd.read_file(shapefile_path)
gdf[“spopt_v244”] = pd.to_numeric(gdf[“spopt_v244”], errors=“coerce”)
gdf[“spopt_v244”] = gdf[“spopt_v244”].fillna(gdf[“spopt_v244”].median())
print(gdf.crs)
gdf = gdf.to_crs(“EPSG:2272”)
````

### Step 3: Define parameters
```python
attrs_name = [“spopt_v244”] # variable we are using to regionalize
w = libpysal.weights.Queen.from_dataframe(gdf) # create spatial weights object from shp
n_clusters = 12 # number of contigous regions we want
floor = 1 # minimum number of spatial objects in each region
trace = False # wether or not we store intermediate values
islands = “increase” # adds “islands” to existing regions, rather than making them their own
````
### Step 4: create minimum cost spanning forest (MCST)
```python
spanning_forest_kwds = dict(
dissimilarity=skm.manhattan_distances,
affinity=None,
reduction=numpy.sum,
center=numpy.mean,
verbose=2
)
````
### Step 5: Solve the model
```python
model = spopt.region.Skater(
gdf,
w,
attrs_name,
n_clusters=n_clusters,
floor=floor,
trace=trace,
islands=islands,
spanning_forest_kwds=spanning_forest_kwds
)
model.solve()
````
### Step 6: Add to data frame
```python
gdf[“demo_regions”] = model.labels_
````
### Step 7: Visualize
```python
gdf[“region”] = model.labels_ # Assign the cluster labels to the GeoDataFrame
gdf.plot(column=“region”, categorical=True, legend=True, figsize=(10, 6))
plt.title(“Skater Clustering with Islands as Separate Regions”)
plt.show()
````

# Spopt_Maximal Covering Location Problem (MCLP)
## Siting Convenient Stores for Low-Income Communities
## Project Description
This project applies the Maximal Covering Location Problem (MCLP) to optimize the placement of grocery stores in a low-income neighborhood. By analyzing household locations and predefined candidate grocery store sites, the model identifies the best locations to maximize the number of households served within a specific service radius (e.g., 5 minutes walking or 300 meters).

Using a combination of real-world distance data from OSRM and the open-source CBC solver, the project provides actionable insights for equitable and efficient urban planning, focusing on improving food accessibility for underserved communities.

### Key Features A. 
**Demand Points** : Locations that require service, such as households or population centers. Represented as coordinates or geographic datasets.

**Candidate facility locations** : Predefined potential locations where facilities (e.g., grocery stores) can be sited. Represented as coordinates or geographic datasets.

**Cost Matrix** : A matrix representing the cost (e.g., travel time or distance) between each demand point and each candidate facility location.  Computed using the OSRM  routing engine.

**Service Radius** : The maximum acceptable distance or travel time within which a demand point is considered "covered" by a facility. **The defaults are meters or seconds**. 
**Number of facilities to site** : The total number of facilities that can be selected in the solution.

**Demand Weights** : Optional weights assigned to each demand point, reflecting its importance or priority.

### Key Features B. 
**Solver : pulp.COIN_CMD** : The solver in in this script used the  pulp.COIN_CMD. its specifies the optimization engine that solves the mathematical problem formulated by the MCLP model. Most windows comes with it, but in this tutorial I have to download and extracted to a specific location in “C:/” drive. It can be downloaded here:   [https://github.com/coin-or/Cbc/releases](URL)

**(OSRM) Open Source Routing Machine**: Instead of using straight-line (Euclidean) distances, OSRM ensures travel costs are based on real-world road networks, making the results more realistic. [https://router.project-osrm.org](URL) OSRM provides accurate travel-time or distance matrices between demand points (households) and candidate facilities (e.g., grocery stores).

## Case Study
Imagine being tasked by a city engineer to strategically choose three grocery store locations from a list of ten predefined options in a low-income neighborhood. The challenge: how do you ensure these stores are placed to maximize accessibility for households, all while staying within a specific service radius, such as 300 meters or 5 minutes walking distance? The focus isn’t just on logistics but on fairness—how do you ensure equitable coverage so that as many households as possible benefit? This scenario pushes you to think critically about resource allocation, urban planning, and addressing food accessibility gaps in underserved communities. 

### Workflow
#### Prerequitise
***Intall these libraries in your local python environment if tou hva not, see the instructions in the "pre-instruction file before running this script using "pip" when necessary. 
***
pulp     : 2.8.0
spopt    : 0.5.1.dev59+g343ef27
shapely  : 2.0.6
routingpy: 1.2.1
folium   : 0.18.0
geopandas: 1.0.1
numpy    : 1.26.2
 
### 1.0 Importing Necessary Packages

## Importing Necessary Packages

```python
# Importing necessary packages
import geopandas as gpd
import folium
import pandas as pd
from shapely.geometry import Point
from spopt.locate import simulated_geo_points
import pulp
import routingpy
from routingpy import OSRM
import shapely
from shapely.geometry import shape, Point
import spopt
import numpy as np
import matplotlib.pyplot as plt
from spopt.locate import MCLP
````


### 2.0 Load shapefile of the region of interest
The original documentation used json file, but in this tutorial I used a shapefile, based on the results with a regional analysis performed with *******.

## Load and Visualize Areas of Interest

```python
# Load areas of interest (shapefile)
pf_region = "C://pf_region.shp"

#read as as Geodataframe(gdf)
gdf = gpd.read_file(pf_region)

# Visualize the low-income region
gdf.plot()
plt.show()

print(gdf.crs)

````
#### The map below shows low income tracts, in Philadelphia 
![Area of Interest in Philadelphia](https://github.com/niiquaye70/Spopt_MCLP/blob/main/Area%20of%20interest_Phily.png)

### 3.0 Adding Demand Points (Households) and Visualise using folium 

The next step is to generate the target households locations within the defined area of interest and extract their coordinates for further analysis. 
Using the ***simulated_geo_points function***, the households are randomly placed within the geographic boundaries of the region (gdf). These points are then converted into a list of [longitude, latitude] coordinates for easier processing and integration into further calculations. ***NB*** Also We need to make sure the CRS is in the geograpjhic coordinate system and as such we print the first five. 

```python
NUM_HOUSEHOLDS = 200

# Generate the households as random points within the region
households = simulated_geo_points(gdf, needed=NUM_HOUSEHOLDS, seed=0)

# Convert households to a list of coordinates
households_coords = households.geometry.map(lambda pt: [pt.x, pt.y]).to_list()

# Check the first few coordinates
print(households_coords[:5])
````
![Coordinates format](https://github.com/niiquaye70/Spopt_MCLP/blob/main/coord_nates%20verify.png)

#### 3.1 Creating a Center of the Region of Interest 
Moving on, we calculate the  calculates the center of the geographic region to use as the initial view for an interactive map. This calculated center point is essential for centering the map appropriately, ensuring the region of interest is fully visible when visualizing the data. 

***gdf.geometry.centroid***: Calculates the centroid of each geometry in the GeoDataFrame.
***y.mean() and x.mean()***: Computes the mean latitude and longitude of all centroids to determine the central point of the map
```python
# Create a base map centered on the geometry
region_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
````
#### 3.2 Creating a Basemap and Visualise households within the Region of Interest 
Moving on we use folium.Map to creates an interactive map centered on the defined geographic region and overlays the region's boundary. 
zoom_start=14 sets the initial zoom level for the map.
tiles="cartodbpositron" applies a clean and modern tile style for visualization.

```python
# Create a base map centered on the geometry
region_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
m = folium.Map(location=region_center, zoom_start=14, tiles="cartodbpositron")

# Convert the geodataframe's geometry to GeoJSON format. 
folium.GeoJson(
    data=gdf.geometry.to_json(),
    name="Region Boundary",
    style_function=lambda x: {"color": "blue", "fillColor": "lightblue"}
).add_to(m)
````
![Base Map](https://github.com/niiquaye70/Spopt_MCLP/blob/main/Base_Map.png)

#### 3.2.1 Adding Household Locations to the interactuve map 
We also need to  add the household locations to the interactive map as points to represent demand points visually. The for loop Iterates through the list of household coordinates. 

```python
# Add households as blue points
for coord in households_coords:
    folium.CircleMarker(
        location=[coord[1], coord[0]],
        radius=3,
        fill=True,
        fill_opacity=1,
        color="cornflowerblue",
        popup="Household"
    ).add_to(m)
````

#### 3.2.2  Save and display the map
```python
# Save and display the map
m.save("households_map.html")
m
````
![Households Map](https://github.com/niiquaye70/Spopt_MCLP/blob/main/Households_.png)

### 3.3 Facility Locations 
We are going to add the facility locations, in this case we used a csv with some predetermined facility locations. 
#### 3.3.1 
This code loads a CSV file containing grocery store data and checks its structure to ensure the latitude and longitude columns are presen and are in the correct format for the model. 
```python
# Load the CSV file
grocery_df = pd.read_csv("C:/GUS5031/Restaurant/grocery_store.csv")

# Ensure the latitude and longitude columns are present
print(grocery_df.head())
````
![Structure of grocery stores](https://github.com/niiquaye70/Spopt_MCLP/blob/main/facility_head.png)

#### 3.3.2 Convert the Data frame into a Geodataframe
Here, this code converts the grocery store data from a Pandas DataFrame into a GeoDataFrame for spatial analysis and checks its coordinate reference system (CRS).
```python
# Create a GeoDataFrame with the projected CRS (e.g., EPSG:2272)
groceries_gdf = gpd.GeoDataFrame(
    grocery_df,
    geometry=gpd.points_from_xy(grocery_df['POINT_X'], grocery_df['POINT_Y']),
    crs="EPSG:2272"  # Projected CRS (update if necessary)
)

# Check the CRS
print("Original CRS:", groceries_gdf.crs)
````
#### 3.3.3 Reproject Into Geographic  CRS
Re-projects the data from its original projected CRS (e.g., EPSG:2272) to the geographic CRS EPSG:4326, commonly used for latitude and longitude. Geographic CRS is required for mapping tools like Folium. 

```python
# Re-project to geographic CRS (EPSG:4326)
groceries_gdf = groceries_gdf.to_crs("EPSG:4326")
print("Re-projected CRS:", groceries_gdf.crs)
print(groceries_gdf.head())
````
#### 3.3.4 Extract Coordinates for plotting
This step extracts the latitude and longitude coordinates of grocery store locations from the GeoDataFrame for use in mapping and analysis. Using the geometry.map function, each geometric point is converted into a list of ***[longitude, latitude]*** pairs. These coordinates are stored in the grocery_coords variable as a simple list, making it easier to integrate with mapping tools like Folium. Finally, the first few coordinates are printed to verify the conversion process and ensure the data is correctly formatted
```python
# Extract grocery coordinates as latitude and longitude
grocery_coords = groceries_gdf.geometry.map(lambda pt: [pt.x, pt.y]).to_list()
print("First few grocery coordinates:", grocery_coords[:5])
````
[Extracted Coordinates system](https://github.com/niiquaye70/Spopt_MCLP/blob/main/Grocer_coord_projected.png)

#### 3.3.5  Visualisation Facility Locations
This step enhances the interactive map by visually identifying grocery store locations with easily recognizable icons. It allows users to explore the geographic distribution of stores and interact with their details. The resulting map is saved as an HTML file for further use or sharing.
```python
# Add grocery stores to the map as green points
# Add grocery stores to the map with shopping cart icons
for i, coord in enumerate(grocery_coords):
    folium.Marker(
        location=[coord[1], coord[0]],  # Latitude, longitude
        icon=folium.Icon(icon="shopping-cart", prefix="fa", color="green"),
        popup=f"Grocery Store {i}"
    ).add_to(m)

# Save and display the map
m.save("households_and_groceries_map_with_icons.html")
m
````
### 3.4 Comput Distance Matrix 
Merges the lists of household coordinates (households_coords) and grocery store coordinates (grocery_coords) into a single list.
This unified list is essential for calculating distances or travel times between all households and grocery stores.
#### 3.4.1 Combine all Locations (Demand and Facility Location)
```python
# Combine all locations (households + grocery stores)
all_locations = households_coords + grocery_coords
````
### 3.4.2 Prepare sources and Destinations 
Routing tools require source and destinations indices to compute the distance or travel time matrix. We will create indices or list  that corresponds to the len of the sources and destinations respectively. This separation allows efficient computation between households (sources) and grocery stores (destinations).By defining source_indices and target_indices, the code ensures households (demand points) and grocery stores (potential facilities) are clearly distinguished within the unified list.
```python
# Define indices for sources (households) and destinations (grocery stores)
source_indices = list(range(len(households_coords)))  # Indices of households
target_indices = list(range(len(households_coords), len(all_locations)))  # Indices of grocery stores
````
#### 3.4.3 Compute distanace matrix for walking
We will now  initializes the OSRM client and uses it to compute the distance matrix (travel times in seconds) between households and grocery stores because we are considering walkability. 
```python
# Initialize the OSRM client
client = OSRM(base_url="https://router.project-osrm.org")

# Compute the distance matrix (travel times in seconds)
osrm_routing_matrix = client.matrix(
    locations=all_locations,
    sources=source_indices,
    destinations=target_indices,
    profile="pedestrian"  # Use pedestrian mode for walking distances
)

````

```python
# Extract durations and convert to a NumPy array
cost_matrix = np.array(osrm_routing_matrix.durations)
print(cost_matrix)
# Check the dimensions of the matrix
# (Number of households, Number of grocery stores)
print("Cost matrix shape:", cost_matrix.shape)  
````
***We used a 2D numpy array to hold the values of travel time for each household. there are 200 households and 13 grocery stores, the matrix will be of shape (200, 10)***

##### Cost Matrix Output and Shape
The following image shows the output of the cost matrix, including its values and shape:
![Cost Matrix and Shape](https://github.com/niiquaye70/Spopt_MCLP/blob/main/matrix_output%20and%20shape.png)

### 3.5 Running the Maximize  Covering Location Problem 
```python
# Set MCLP parameters
SERVICE_RADIUS = 300  # Maximum acceptable service time (e.g., 5 minutes in seconds)
P_FACILITIES = 3      # Number of grocery stores to site
my_weights = np.ones((1, len(households_coords)), dtype=int)  # Equal weights for households
````
#### 3.5 1 Initialize the solver
We initializes the Maximal Covering Location Problem (MCLP) solve to determine the optimal placement of facilities to maximize the coverage of demand points (e.g., households). 
If not locally available download: and extact into local drive: solver = pulp.COIN_CMD(path="******cbc.exe", msg=False)
```python
# Initialize and solve the MCLP with parameters 
solver = pulp.COIN_CMD(path= "C:/cbc/bin/cbc.exe",msg=False)
mclp = MCLP.from_cost_matrix(cost_matrix, wght , SERVICE_RADIUS, p_facilities=P_FACILITIES)
mclp = mclp.solve(solver)
````
#### 3.5.2  Retrieving and Printing Results : objective value and percentage coverage 
We are first going to retrieve the objective value, thus the total number of households as an integer. In addition we will also determine the percentage coverage, this helps to access the efficiency of the solution we are providing. 

```python
# Print results
obj_val = int(mclp.problem.objective.value())
obj_perc = mclp.perc_cov

#Evaluated the results 
print(
    f"Of the {len(households_coords)} households, {obj_val} have access to at least 1 "
    f"grocery store within a {SERVICE_RADIUS // 60}-minute walk. This is {obj_perc:.1%} coverage."
)

````
If every works well you termina should print: ****Of the 200 households, 173 have access to at least 1 grocery store within a 5-minute walk. This is 8650.0% coverage****

### 4. 0 Visualisation 
Create feature groups in Folium map for presentation 
```python
# Create feature groups for sited grocery stores and allocated households
sited_groceries_fg = folium.FeatureGroup("Sited Grocery Stores").add_to(m)
allocated_households_fg = folium.FeatureGroup("Allocated Households").add_to(m)
````
Initiaze a list of colors 

# Use distinct colors for visualization
colors = ["darkcyan", "saddlebrown", "coral", "gold", "purple", "blue", "lime", "orange", "pink"]

````

#### Adding model results to folium
Here we visualizes the sited grocery stores and the households they serve on the Folium map by adding markers for each grocery store and circle markers for allocated households.
The code performs an iteration on the range of coordinates in the "grocery_coords" varaible, the mclp.fac2cli is an attribute of the MCLP that represent the assiggnment of the households to the sited stores.
```python

for i in range(len(grocery_coords)):
    if mclp.fac2cli[i]:  # Check if the grocery was selected
        folium.Marker(
            location=[grocery_coords[i][1], grocery_coords[i][0]],
            icon=folium.Icon(
                icon="shopping-cart",
                prefix="fa",
                color="red"
            ),
            popup=f"Sited Grocery {i}"
        ).add_to(sited_groceries_fg)

        for j in mclp.fac2cli[i]:  # Allocated households
            folium.CircleMarker(
                location=[households_coords[j][1], households_coords[j][0]],
                radius=10,
                fill=True,
                color=colors[i % len(colors)]
            ).add_to(allocated_households_fg)
# Add a layer control and save the map
folium.LayerControl().add_to(m)
````
### Save and Visualize
```python
m.save("final_solution_map.html")
````
![Solution Map](https://github.com/niiquaye70/Spopt_MCLP/blob/main/Solution_map.png)

## Exercise 
A city planner wants to identify the optimal locations for two ***Bus shelters*** to maximize the number of ***parks and recreation area*** that can access them within a 10-minute drive. Using real-world travel times from OSRM (using the "car" profile), the goal is to ensure the most equitable access for households while minimizing driving time.
##### Instructions 
1. **Download the following data set: Bus shelters, parks and recreation and the region of interest**
2. **Load and Visualize the Data**:
  - Load the region of interest shapefile.
  - Load the parks and recreation dataset as a GeoDataFrame.
  - Load the bus shelters dataset as a GeoDataFrame
3. **Prepare Data**
  - Re-project datasets to a geographic CRS (e.g., EPSG:4326) if necessary.
  - Combine coordinates for parks (sources) and bus shelters (destinations)
4. **Set matrix profile to "car"**
5.**Solve the MCLP** 
  - Set the service radius to 600 seconds (10-minute drive)
  - Optimize the placement of two bus shelters
  - Create your own list of colors
  - Print the outcome of the model using the objective value and coverage percentage

## Reference
Timothy Ellersiek, James Gaboardi() Siting Restaurants with MCLP in San Francisco
RM Assunção, MC Neves, G Câmara, and C da Costa Freitas. Efficient regionalization techniques for socio-economic geographical units using minimum spanning trees. International Journal of Geographical Information Science, 20(7):797–811, 2006. doi:10.1080/13658810600665111.
Richard L. Church and Alan T. Murray. Business Site Selection, Locational Analysis, and GIS. John Wiley & Sons, Inc., Hoboken, 2009.
OpenDataPhily, https://opendataphilly.org/datasets/neighborhood-food-retail/
