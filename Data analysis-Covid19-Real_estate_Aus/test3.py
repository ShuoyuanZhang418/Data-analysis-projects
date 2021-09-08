import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import folium
import webbrowser
from mpl_toolkits.basemap import Basemap

#import os
#os.environ['PROJ_LIB'] = r'D:\Anaconda\Library\share\basemap'

import warnings
warnings.filterwarnings("ignore")

# Read data from csv
file_data = pd.read_csv('completed_aus-property-sales-sep2018-april2020.csv')
print('Length: {} rows.'.format(len(file_data)))
print(file_data.head())

print(file_data.describe())


# matplotlib inline
LON_MIN, LON_MAX = 110, 155
LAT_MIN, LAT_MAX = -39, -30

plt.figure(figsize=[15, 15])
with plt.style.context(('dark_background')):
    plt.scatter(file_data["lon"], file_data["lat"], color='yellow', s=0.1, alpha=0.1)
    # outliers are making range-estimation poor, so let's limit it manually
    plt.xlim([LON_MIN, LON_MAX])
    plt.ylim([LAT_MIN, LAT_MAX])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    m = Basemap()
    m.drawcoastlines()
    plt.show()


# get a sample data in the file data
data = file_data.sample(frac=0.001, random_state=12, axis=0)

# Instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# Loop through the 200 crimes and add each to the incidents feature group
for lat, long, in zip(data.lat, data.lon):
    incidents.add_child(
        folium.CircleMarker(
            [lat, long],
            radius=1,
            color='yellow',
            fill_color='red',
            fill_opacity=0.4
        )
    )

# Add incidents to map
national_map = folium.Map(location=[-25, 150], zoom_start=5)
national_map.add_child(incidents)
file_path = r"C:/Users/18811/Desktop/UQ 2021S1课程资料/DATA7001/GROUP ass/Image/map.html"
national_map.save(file_path)
#webbrowser.open(file_path)