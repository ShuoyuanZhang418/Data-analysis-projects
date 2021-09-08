import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# construct Label Encoder
lb = LabelEncoder()

# Read data from csv
file_data = pd.read_csv('data/completed_aus-property-sales-sep2018-april2020.csv')
print('Length: {} rows.'.format(len(file_data)))
print(file_data.head())

# Deepcopy the file data
file_data_copy = copy.deepcopy(file_data)

# Use LabelEncoder to transform the property_type
# file_data_copy[["property_type"]] = lb.fit_transform(file_data_copy["property_type"])
file_data_copy[["suburb"]] = lb.fit_transform(file_data_copy["suburb"])

# Use One-hot Encoder to transform the property_type and suburb
file_data_copy = file_data_copy.join(pd.get_dummies(file_data_copy.property_type))
file_data_copy = file_data_copy.join(pd.get_dummies(file_data_copy.city_name))

# Drop ' 00:00:00' and then split data to year, month and day
date_sold = file_data_copy["date_sold"].str.replace(' 00:00:00', '').str.split('-')
file_data_copy["date_sold_year"] = date_sold.map(lambda x: x[0]).astype(int)
file_data_copy["date_sold_month"] = 0.1 * date_sold.map(lambda x: x[1]).astype(int)
file_data_copy["date_sold_day"] = 0.01 * date_sold.map(lambda x: x[2]).astype(int)
print(file_data_copy.head())


def cross_validation(x, y, model):
    kf = KFold(n_splits=10, shuffle=False)
    kf.get_n_splits(x)
    err_list = []
    for train_index, test_index in kf.split(x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit the model
        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)
        loss = metrics.mean_squared_error(y_test, y_pred)
        err_list.append(loss)
    return np.mean(err_list)


def model_selection(x, y):
    ##########################################################################
    model = LinearRegression()
    err = cross_validation(x, y, model)
    print("------------------------------------------------")
    print("Linear regression cross validation error = ", err)
    print("------------------------------------------------")
    ##########################################################################
    model = RandomForestRegressor(n_estimators=500, random_state=0)
    err = cross_validation(x, y, model)
    print("------------------------------------------------")
    print("RF regression cross validation error = ", err)
    print("------------------------------------------------")
    ##########################################################################
    model = KNeighborsRegressor(n_neighbors=30)
    err = cross_validation(x, y, model)
    print("------------------------------------------------")
    print("KNN regression cross validation error = ", np.mean(err))
    print("------------------------------------------------")
    ##########################################################################
    model = MLPRegressor(max_iter=2000)
    err = cross_validation(x, y, model)
    print("------------------------------------------------")
    print("NN regression cross validation error = ", np.mean(err))
    print("------------------------------------------------")


# colums that need to be removed
remove_col = ["price", "property_type", "Unnamed: 0", "lat", "lon", "date_sold_int", "date_sold", "city_name", "state",
              "loc_pid", "lga_pid"]
file_data_copy["date_sold_int"] = date_sold.map(lambda x: x[0] + x[1] + x[2]).astype(int)

file_data_copy_sample = file_data_copy.sample(frac=0.2, random_state=12, axis=0)
X_sample = file_data_copy_sample.drop(remove_col, axis=1)
Y_sample = file_data_copy_sample["price"]

# This part is going to take a lot of time to run.
# model_selection(X_sample, Y_sample)

# We can observed from the result is that xxx has the best performance.
# Train the model with this model
# RF_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=0, verbose=2)
RF_model = RandomForestRegressor(verbose=2)
X_train = file_data_copy[(file_data_copy["date_sold_int"] < 20200101)].drop(remove_col, axis=1)
# print(X_train)
normalise = MinMaxScaler()
X_train = pd.DataFrame(normalise.fit_transform(X_train))
Y_train = file_data_copy["price"][(file_data_copy["date_sold_int"] < 20200101)]
RF_model.fit(X_train, Y_train)
print(RF_model.feature_importances_)

# Predict
X_fit = file_data_copy[(file_data_copy["date_sold_int"] >= 20200101)].drop(remove_col, axis=1)
X_fit = pd.DataFrame(normalise.fit_transform(X_fit))
Y_fit = RF_model.predict(X_fit)
file_data_copy["price"][(file_data_copy["date_sold_int"] >= 20200101)] = np.round(Y_fit)

# Write the completed data into a csv file
# file_data_copy.to_csv("data/predicted_aus-property-sales-sep2018-april2020.csv")
file_data_copy.to_csv("data/3-predicted_aus-property-sales-sep2018-april2020.csv")
