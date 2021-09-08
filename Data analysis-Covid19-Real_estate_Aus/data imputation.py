import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# construct Label Encoder
lb = LabelEncoder()

# Read data from csv
data_path_property = 'data/aus-property-sales-sep2018-april2020.csv'
data_path_covid = 'data/owid-covid-data.csv'
file_data_property = pd.read_csv(data_path_property)
file_data_covid = pd.read_csv(data_path_covid)

print('Property data: Length: {} rows.'.format(len(file_data_property)))
print(file_data_property.head())

print('Covid data: Length: {} rows.'.format(len(file_data_covid)))
print(file_data_covid.head())

# Show data types of each column
print(file_data_property.dtypes)
print(file_data_covid.dtypes)

# Check the number of NULL in each column
# We can observed that there are a huge number of NaN in price column
print(file_data_property.isnull().sum())
print(file_data_covid[file_data_covid["location"] == "Australia"].isnull().sum())

# Replace NaN with 0
file_data_property.fillna(0, inplace=True)

# Directly drop rows which lat and lon is 0 (because there is only 78 rows that contains NaN)
row_list = file_data_property[file_data_property.lat == 0].index.tolist()
file_data_property = file_data_property.drop(row_list)
print(file_data_property.head())

# Deepcopy the file data
file_data_property_copy = copy.deepcopy(file_data_property)

# construct boxplots of each columns to detect outliers.
pd.DataFrame({'log(price)': np.log10(file_data_property_copy["price"][file_data_property_copy["price"] != 0])}).boxplot()
plt.title('Boxplot of log(price)')
plt.show()

pd.DataFrame({'total cases': file_data_covid["total_cases"][file_data_covid["location"] == "Australia"]}).boxplot()
plt.title('Boxplot of total cases')
plt.show()

pd.DataFrame({"new cases":file_data_covid["new_cases"][file_data_covid["location"] == "Australia"]}).boxplot()
plt.title('Boxplot of new cases')
plt.show()

pd.DataFrame(file_data_property_copy["price"][file_data_property_copy["price"] != 0]).boxplot()
plt.title('Boxplot of price')
plt.show()

pd.DataFrame({'latitude': file_data_property_copy["lat"]}).boxplot()
plt.title('Boxplot of latitude')
plt.show()

pd.DataFrame({'longitude': file_data_property_copy["lon"]}).boxplot()
plt.title('Boxplot of longitude')
plt.show()

pd.DataFrame(file_data_property_copy["bedrooms"]).boxplot()
plt.title('Boxplot of bedrooms')
plt.show()

# Drop the extremely large outlier
file_data_property_copy = file_data_property_copy.drop(file_data_property_copy
                                                       [(file_data_property_copy["price"] == 100000000)].index)

file_data_property_copy = file_data_property_copy.drop(file_data_property_copy
                                                       [(file_data_property_copy["price"] > 0) &
                                                        (file_data_property_copy["price"] <= 10000)].index)
# See again
pd.DataFrame(np.log10(file_data_property_copy["price"][file_data_property_copy["price"] != 0])).boxplot()
plt.title('Boxplot of price')
plt.show()

# print the number of records with different threshold and chose one.
print(len(file_data_property_copy[(file_data_property_copy["price"] >= 2500000)]))

# Filter the price by the threshold
file_data_property_copy = file_data_property_copy.drop(file_data_property_copy
                                                       [(file_data_property_copy["price"] >= 2500000)
                                                        | (file_data_property_copy["price"] < 50000)
                                                        & (file_data_property_copy["price"] != 0)
                                                        | (file_data_property_copy["lat"] >= 0)].index)
file_data_property = file_data_property.drop(file_data_property[(file_data_property["lat"] >= 0)].index)
# Drop ' 00:00:00' and then split data to year, month and day
file_data_property_copy["date_sold"] = file_data_property_copy["date_sold"].str.replace(' 00:00:00', '').str.split('-')
file_data_property_copy["date_sold_year"] = file_data_property_copy["date_sold"].map(lambda x: x[0]).astype(int)
file_data_property_copy["date_sold_month"] = 0.1 * file_data_property_copy["date_sold"].map(lambda x: x[1]).astype(int)
file_data_property_copy["date_sold_day"] = 0.01 * file_data_property_copy["date_sold"].map(lambda x: x[2]).astype(int)

# Since the latitude and longitude difference is small, multiply them by 10
file_data_property_copy["lat"] = 10 * file_data_property_copy["lat"]
file_data_property_copy["lon"] = 10 * file_data_property_copy["lon"]

# Use LabelEncoder to transform the property_type
file_data_property_copy[["property_type"]] = lb.fit_transform(file_data_property_copy["property_type"])


def best_k_cross_validation():
    # Perform a cross-validation to chose the best k of knn(This part is going to take a lot of time to run because
    # there are a lot of possible values of k)

    # Sample from the file data
    file_data_copy_sample = file_data_property_copy.sample(frac=0.2, random_state=12, axis=0)
    X = file_data_copy_sample[["lat", "lon", "bedrooms", "property_type", "date_sold_year", "date_sold_month",
                               "date_sold_day"]][file_data_property["price"] != 0]
    Y = file_data_copy_sample["price"][file_data_property["price"] != 0]

    # Possible values of k
    k_arr = np.arange(1, 50, 1)

    # Cross-validation Scores of different k
    scores = list()
    err_list = []

    # Construct a k-fold
    kf = KFold(n_splits=5, shuffle=False)
    kf.get_n_splits(X)

    # Perform cross-validation of each k in k_arr
    for k in k_arr:
        err_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            knn = KNeighborsRegressor(k, weights='distance')
            knn.fit(X_train, y_train)
            y_val_pred = knn.predict(X_test)
            loss = np.mean(np.power(y_val_pred - y_test, 2))
            err_list.append(loss)
        err = np.mean(err_list)
        scores.append(err)

    # Plot the validation error as a function of the number of neighbors k to
    # determine the best k. Use the matplotlib library.
    plt.plot(k_arr, scores)
    plt.ylabel('CV score (mean squared error)')
    plt.xlabel('k - # of neighbours')
    plt.title("MSE for Different K")
    plt.show()

    print("The best k is", np.argmin(scores))


def knn_imputation():
    # Use knn regression to impute the price column

    # Construct X and Y for training
    X = file_data_property_copy[["lat", "lon", "bedrooms", "property_type", "date_sold_year", "date_sold_month",
                                 "date_sold_day"]][file_data_property["price"] != 0]
    Y = file_data_property_copy["price"][file_data_property["price"] != 0]

    x_fit = file_data_property_copy[["lat", "lon", "bedrooms", "property_type", "date_sold_year", "date_sold_month",
                                     "date_sold_day"]][file_data_property["price"] == 0]

    # Use the best k value obtained from cross-validation
    k = 36

    # Use KNeighborsRegressor() to train the data
    clf = KNeighborsRegressor(n_neighbors=k, weights="distance")
    clf.fit(X, Y)

    # Fit the data using constructed knn model
    y_fit = clf.predict(x_fit)
    print(np.mean(y_fit))

    # Fill the missing data in the file
    print(len(file_data_property["price"][file_data_property["price"] == 0]))
    print(len(y_fit))
    file_data_property["price"][file_data_property["price"] == 0] = np.round(y_fit)
    # file_data["date_sold_year"] = file_data_copy["date_sold_year"]
    # file_data["date_sold_month"] = file_data_copy["date_sold_month"]
    # file_data["date_sold_day"] = file_data_copy["date_sold_day"]
    print(file_data_property)
    completed_file_data = file_data_property.drop(file_data_property[(file_data_property["price"] == 100000000)].index)
    completed_file_data = completed_file_data.drop(completed_file_data
                                                           [(completed_file_data["price"] > 0) &
                                                            (completed_file_data["price"] <= 10000)].index)
    completed_file_data = completed_file_data.drop(file_data_property[(file_data_property["lat"] >= 0)].index)

    # Write the completed data into a csv file
    completed_file_data.to_csv("data/completed_aus-property-sales-sep2018-april2020.csv")


if __name__ == '__main__':
    knn_imputation()
    #best_k_cross_validation()