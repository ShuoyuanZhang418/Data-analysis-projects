import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
#import folium
import webbrowser
from mpl_toolkits.basemap import Basemap

#import os
#os.environ['BASEMAPDATA'] = r'D:\Anaconda\Library\share\basemap'

import warnings
warnings.filterwarnings("ignore")

# Read data from csv
file_data = pd.read_csv('completed_aus-property-sales-sep2018-april2020.csv')
print('Length: {} rows.'.format(len(file_data)))
print(file_data.head())

print(file_data.describe())

# Deepcopy the file data
file_data_copy = copy.deepcopy(file_data)

# Filter
file_data_copy_1 = file_data_copy[file_data_copy["city_name"] == "Sydney"]
# file_data_copy_1 = file_data_copy[file_data_copy["price"] > 2000000]

# matplotlib inline
LON_MIN, LON_MAX = 110, 155
LAT_MIN, LAT_MAX = -39, -30

plt.figure(figsize=[15, 15])
#with plt.style.context(('dark_background')):
# outliers are making range-estimation poor, so let's limit it manually
plt.scatter(file_data_copy_1["lon"], file_data_copy_1["lat"], color='red', s=0.1, alpha=0.1)
plt.xlim([LON_MIN, LON_MAX])
plt.ylim([LAT_MIN, LAT_MAX])
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Australia map
national_map = Basemap(llcrnrlon=110, llcrnrlat=-39, urcrnrlon=155, urcrnrlat=-10, resolution='i', lat_0=-35, lon_0=128)

# Brisbane map
#national_map = Basemap(llcrnrlon=152.5, llcrnrlat=-28, urcrnrlon=153.5, urcrnrlat=-27, resolution='i', lat_0=-35, lon_0=128)

# Sydney map
#national_map = Basemap(llcrnrlon=150, llcrnrlat=-34.5, urcrnrlon=152, urcrnrlat=-33, resolution='i', lat_0=-35, lon_0=128)

# Canberra map
#national_map = Basemap(llcrnrlon=148.5, llcrnrlat=-36, urcrnrlon=150, urcrnrlat=-34.5, resolution='i', lat_0=-35, lon_0=128)

# Melbourne map
#national_map = Basemap(llcrnrlon=144, llcrnrlat=-39, urcrnrlon=146, urcrnrlat=-37, resolution='i', lat_0=-35, lon_0=128)

# Adelaide map
#national_map = Basemap(llcrnrlon=137.6, llcrnrlat=-36, urcrnrlon=139.5, urcrnrlat=-33.5, resolution='i', lat_0=-35, lon_0=128)

# Perth map
#national_map = Basemap(llcrnrlon=115, llcrnrlat=-32.8, urcrnrlon=116.5, urcrnrlat=-31, resolution='i', lat_0=-35, lon_0=128)

national_map.drawcoastlines()
national_map.readshapefile("gadm36_AUS_shp/gadm36_AUS_0",  'gadm36_AUS_0', drawbounds=True)
national_map.readshapefile("gadm36_AUS_shp/gadm36_AUS_1",  'gadm36_AUS_1', drawbounds=True)
national_map.readshapefile("gadm36_AUS_shp/gadm36_AUS_2",  'gadm36_AUS_2', drawbounds=True)
longitudes, latitudes = national_map(file_data_copy_1["lon"], file_data_copy_1["lat"])
national_map.scatter(longitudes, latitudes, s=1, marker='*', facecolors='r', edgecolors='r')
plt.show()
