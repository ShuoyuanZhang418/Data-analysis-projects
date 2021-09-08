# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PREAMBLE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
Linear Regression
Regression
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LOAD PACKAGES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SET PARAMETERS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set seed for reproducibility
random.seed(1)

# Set the proportion of the dataset that will be used for the training set
prop_train = 0.8

# Set whether a validation set will be used and the split between validation and test sets
# (i.e. 0.7 means 70% of remaining data after training set is made will be put into validation and 30% to test)
# If validation set is used, set whether or not to assess the model on the test set
# (use once model selected from validation set runs)
use_validation_set, prop_validation, assess_test_set = False, 0.5, False

# Set whether to use cross-validation and how many folds, k, if using it
use_CV, CV_folds = False, 5

# Set whether to use PCA and if so, how many principal components to keep
use_PCA, num_PCs = False, 2

# Set whether to use scaling on the data
use_scaling = False

# Set whether to normalise the data (between 0 and 1)
use_normalisation = False

# Set the column name that contains the response variable, Y
response_col = "price"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SET HYPERPARAMETERS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Choose whether the intercept can be non-zero (don't fit it to 0)
nonzero_intercept = True

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LOAD DATA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = pd.read_csv("data/Matt_ML_Data.csv")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FEATURE ENGINEERING & VARIABLE REMOVAL
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Choose which columns to keep
cols_to_remove = ["Unnamed: 0", "date_sold", "lat", "lon",  # "bedrooms", "property_type"
                  "loc_pid", "lga_pid", "suburb"]
data = data.drop(cols_to_remove, axis=1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ONE-HOT ENCODING
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Specify the column names that require one-hot encoding
features_to_encode = [data.columns[i] for i in range(len(data.columns))
                      if data[data.columns[i]].dtype != np.number
                      and data.columns[i] != response_col]


def encode_and_bind(data, feature_to_encode):
    dummies = pd.get_dummies(data[[feature_to_encode]])
    new_data = pd.concat([data, dummies], axis=1)
    new_data = new_data.drop([feature_to_encode], axis=1)
    return (new_data)


for feature in features_to_encode:
    data = encode_and_bind(data, feature)

# Remove columns that only contain 1 value
cols_to_remove = []
for j in range(len(data.columns)):
    if len(data[data.columns[j]].unique()) == 1:
        cols_to_remove.append(data.columns[j])
    else:
        pass

data = data.drop(cols_to_remove, axis=1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SPLIT DATA INTO FEATURES, X, AND RESPONSES, Y
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Split the data into features and responses
X = data.drop(response_col, axis=1)
Y = data[response_col]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DO PCA ON DATA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if use_PCA == True:
    pca = PCA(n_components=num_PCs)
    X = pd.DataFrame(pca.fit_transform(X))
else:
    pass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SCALE FEATURES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if use_scaling == True:
    X = pd.DataFrame(scale(X))
else:
    pass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NORMALISE FEATURES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if use_normalisation == True:
    normalise = MinMaxScaler()
    X = pd.DataFrame(normalise.fit_transform(X))
else:
    pass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# USE CROSS-VALIDATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create empty vectors for results
error_R2, error_RMSE, error_MAE = [], [], []

# Specify the number of folds of the cross-validation
if use_CV == False:
    CV_folds = 1
else:
    pass

# Generate the indices used for the cross-validation, if it is used
if CV_folds != 1:
    CV_generator = KFold(n_splits=CV_folds, shuffle=True, random_state=1)

    CV_train_indices = []
    CV_test_indices = []
    for train_indices, test_indices in CV_generator.split(X):
        CV_train_indices.append(train_indices)
        CV_test_indices.append(test_indices)

else:
    pass

for CV_k in range(CV_folds):

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SPLIT DATA INTO TRAIN AND TEST SETS
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # If cross-validation used, perform k-fold CV. If not used, split data into specified proportion of test data
    if CV_folds != 1:
        train_indices = CV_train_indices[CV_k]
        test_indices = CV_test_indices[CV_k]

        X_train = X.loc[train_indices, :]
        Y_train = Y[train_indices]
        X_test = X.loc[test_indices, :]
        Y_test = Y[test_indices]

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - prop_train, random_state=1)

    # Create split for data dependent on whether validation set used or not
    if use_validation_set == True:
        X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=1 - prop_validation,
                                                                      random_state=1)

    else:
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # FIT MODEL
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model = LinearRegression(
        fit_intercept=nonzero_intercept
    )
    model = model.fit(X_train, Y_train)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ASSESS ACCURACY
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Assess model accuracy on validation set if specified. Once satisfactory model is fitted,
    # test model on the test set
    if use_validation_set == True and assess_test_set == False:
        pred_Y_validation = model.predict(X_validation)
        error_R2.append(r2_score(Y_validation, pred_Y_validation))
        error_RMSE.append(mean_squared_error(Y_validation, pred_Y_validation, squared=False))
        error_MAE.append(mean_absolute_error(Y_validation, pred_Y_validation))

    else:
        pred_Y_test = model.predict(X_test)
        error_R2.append(r2_score(Y_test, pred_Y_test))
        error_RMSE.append(mean_squared_error(Y_test, pred_Y_test, squared=False))
        error_MAE.append(mean_absolute_error(Y_test, pred_Y_test))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PRINT RESULTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate 95% confidence intervals if CV is used
if use_CV == True:
    error_R2_lower95 = np.mean(error_R2) - 1.96 * np.std(error_R2) / np.sqrt(CV_folds)
    error_R2_upper95 = np.mean(error_R2) + 1.96 * np.std(error_R2) / np.sqrt(CV_folds)
    error_RMSE_lower95 = np.mean(error_RMSE) - 1.96 * np.std(error_RMSE) / np.sqrt(CV_folds)
    error_RMSE_upper95 = np.mean(error_RMSE) + 1.96 * np.std(error_RMSE) / np.sqrt(CV_folds)
    error_MAE_lower95 = np.mean(error_MAE) - 1.96 * np.std(error_MAE) / np.sqrt(CV_folds)
    error_MAE_upper95 = np.mean(error_MAE) + 1.96 * np.std(error_MAE) / np.sqrt(CV_folds)
else:
    pass

# Print results
if use_validation_set == True:
    if use_CV == False:
        print("===== PARAMETERS =====",
              "\nTraining Set Split: ", prop_train * 100, "%",
              "\nValidation Set Split: ", prop_validation * 100, "%",
              "\nAccuracy Assessed on Validation Set: ", not assess_test_set,
              "\nPCA: ", use_PCA,
              "\nScaling: ", use_scaling,
              "\nNormalisation: ", use_normalisation,
              "\n\n===== RESULTS =====",
              "\nAverage R2: ", round(np.mean(error_R2), 3),
              "\nAverage RMSE: ", round(np.mean(error_RMSE), 3),
              "\nAverage MAE: ", round(np.mean(error_MAE), 3),
              sep="")
    else:
        print("===== PARAMETERS =====",
              "\nCross-Validation: ", CV_folds, "-fold",
              "\nValidation Set Split: ", prop_validation * 100, "%",
              "\nAccuracy Assessed on Validation Set: ", not assess_test_set,
              "\nPCA: ", use_PCA,
              "\nScaling: ", use_scaling,
              "\nNormalisation: ", use_normalisation,
              "\n\n===== RESULTS =====",
              "\nAverage R2: ", round(np.mean(error_R2), 3),
              " | 95% Confidence Interval: (", round(error_R2_lower95, 3), ", ", round(error_R2_upper95, 3), ")",
              "\nAverage RMSE: ", round(np.mean(error_RMSE), 3),
              " | 95% Confidence Interval: (", round(error_RMSE_lower95, 3), ", ", round(error_RMSE_upper95, 3), ")",
              "\nAverage MAE: ", round(np.mean(error_MAE), 3),
              " | 95% Confidence Interval: (", round(error_MAE_lower95, 3), ", ", round(error_MAE_upper95, 3), ")",
              sep="")

else:
    if use_CV == False:
        print("===== PARAMETERS =====",
              "\nTraining Set Split: ", prop_train * 100, "%",
              "\nPCA: ", use_PCA,
              "\nScaling: ", use_scaling,
              "\nNormalisation: ", use_normalisation,
              "\n\n===== RESULTS =====",
              "\nAverage R2: ", round(np.mean(error_R2), 3),
              "\nAverage RMSE: ", round(np.mean(error_RMSE), 3),
              "\nAverage MAE: ", round(np.mean(error_MAE), 3),
              sep="")
    else:
        print("===== PARAMETERS =====",
              "\nCross-Validation: ", CV_folds, "-fold",
              "\nPCA: ", use_PCA,
              "\nScaling: ", use_scaling,
              "\nNormalisation: ", use_normalisation,
              "\n\n===== RESULTS =====",
              "\nAverage R2: ", round(np.mean(error_R2), 3),
              " | 95% Confidence Interval: (", round(error_R2_lower95, 3), ", ", round(error_R2_upper95, 3), ")",
              "\nAverage RMSE: ", round(np.mean(error_RMSE), 3),
              " | 95% Confidence Interval: (", round(error_RMSE_lower95, 3), ", ", round(error_RMSE_upper95, 3), ")",
              "\nAverage MAE: ", round(np.mean(error_MAE), 3),
              " | 95% Confidence Interval: (", round(error_MAE_lower95, 3), ", ", round(error_MAE_upper95, 3), ")",
              sep="")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SAVE MODEL
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save model
# pickle.dump(model, open('model.sav', 'wb'))

# Read model
# model = pickle.load(open('model.sav', 'rb'))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ANALYSE MODEL
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model_coefficients = {}
for i in range(len(X.columns)):
    model_coefficients[X.columns[i]] = model.coef_[i]

print("An increase of 1 COVID case in the total cases results in a change in house price by $",
      model_coefficients["total_cases"])
print(model_coefficients)
# Note that we don't really care about the daily cases as houses may not be sold on days where there are many new cases