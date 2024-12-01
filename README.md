
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
***Intall these libraries in your local python environment before running this script using "pip"
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

