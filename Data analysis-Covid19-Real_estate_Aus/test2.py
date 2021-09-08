from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")

lb = LabelEncoder()


# Read data from csv
file_data = pd.read_csv('aus-property-sales-sep2018-april2020.csv')
print('Length: {} rows.'.format(len(file_data)))
file_data.head()

print(file_data.dtypes)
print(file_data.isnull().sum())

print(file_data[["date_sold"]])

file_data.fillna(0, inplace=True)
row_list = file_data[file_data.lat == 0].index.tolist() # 获得含有该值的行的行号
file_data = file_data.drop(row_list)
#print(file_data.head())
file_data_copy = file_data
file_data_copy["date_sold"] = file_data_copy["date_sold"].str.replace(' 00:00:00', '').str.split('-')
file_data_copy["date_sold_year"] = file_data_copy["date_sold"].map(lambda x: x[0]).astype(int)
file_data_copy["date_sold_month"] = file_data_copy["date_sold"].map(lambda x: x[1]).astype(int)
file_data_copy["date_sold_day"] = file_data_copy["date_sold"].map(lambda x: x[2]).astype(int)
print(file_data_copy.dtypes)
file_data_copy[["property_type"]] = lb.fit_transform(file_data["property_type"])
file_data_copy[["bedrooms"]] = file_data["bedrooms"]
print(file_data_copy.head())