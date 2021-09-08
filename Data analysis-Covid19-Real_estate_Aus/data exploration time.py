from datetime import datetime
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Read data from csv
file_data = pd.read_csv('completed_aus-property-sales-sep2018-april2020.csv')


# Deepcopy the file data
file_data_copy = copy.deepcopy(file_data)

file_data["date_sold"] = pd.to_datetime(file_data["date_sold"])

file_data = file_data.drop(file_data[(file_data["price"] == 100000000)].index)
file_data_copy = file_data_copy.drop(file_data_copy[(file_data_copy["price"] == 100000000)].index)

# Conduct a preliminary analysis of the data
#scatter_matrix(file_data_copy, alpha=0.5, c='yellow', diagonal='hist', marker='.', range_padding=0.05)
#plt.show()


plt.scatter(file_data_copy["bedrooms"], file_data_copy["price"], c='yellow', alpha=0.4)
plt.show()

print(file_data.dtypes)
print(file_data["date_sold"])
file_data_1 = file_data.set_index("date_sold")


plt.figure(figsize=(12,6))
file_data_1["price"].plot()
plt.show()

date_sold = file_data_copy["date_sold"].str.replace(' 00:00:00', '').str.split('-')
file_data_copy["date_sold_int"] = date_sold.map(lambda x: x[0] + x[1] + x[2]).astype(int)
print(file_data_copy["date_sold_int"])
#file_data_copy["date_sold_month"] = 0.1 * file_data_copy["date_sold"].map(lambda x: x[1]).astype(int)
#file_data_copy["date_sold_day"] = 0.01 * file_data_copy["date_sold"].map(lambda x: x[2]).astype(int)
print(np.mean(file_data_copy["price"][(file_data_copy["date_sold_int"] < 20200401)]))
print(np.mean(file_data_copy["price"][(file_data_copy["date_sold_int"] >= 20200401)]))
#file_data["price"][file_data["city_name"] == "Sydney"].plot()
#plt.show()