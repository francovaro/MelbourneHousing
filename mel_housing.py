"""
Machine Learning example
from the book "Machine Learning for Aboslute Beginnes" by Oliver Theobald

Francesco Varani
12/2020
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(".\\Melbourne_housing_FULL.csv")

#for colonna in df.columns:
    #print(colonna)

# Scrubbing dataset ! remove unneded features
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Regionname']
del df['Propertycount']
del df['Lattitude']
del df['Longtitude']

print("After scrubbing total columns: " + str(len(df.columns)))
for colonna in df.columns:
    print(colonna)

#removing rows with empty values
df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)

# apply one hot encoding
df = pd.get_dummies(df, columns = ["Suburb", "CouncilArea", "Type"])

# assign X (indipendent var) and y (dependat variables)
X = df.drop('Price', axis = 1)
y = df["Price"]

# now split dataset - training data and test data - 70% train 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

# Select and algorithm and config hyperparameters
model = ensemble.GradientBoostingRegressor(n_estimators = 150, learning_rate = 0.1, max_depth = 30, min_samples_split = 4, min_samples_leaf = 6, max_features = 0.6, loss = "huber")
"""
 we have selected the gradient boosting
 - n_estimators = nr of decision trees
 - learning_reate = rate at which additional trees influence the prediction.
 - max_depth max number of layers for each decision trees
 - min_samples_split = min samples to execute a binary split
 - min_samples_leaf = 
 - max_features =
 - loss = how the model loss is calculated
"""
print("Start training the model")
# train the prediction model
model.fit(X_train, y_train) 
print("Model trained")

# check the result
# we use mean absolute error, passing the data we used to train.
# model.predict runs the model
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set MAE: %.2f" %mae_train)

# let's check with the test data
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set MAE    : %.2f" %mae_test)

# Let's try to optimize
model_2 = ensemble.GradientBoostingRegressor(n_estimators = 250, learning_rate = 0.1, max_depth = 5, min_samples_split = 4, min_samples_leaf = 6, max_features = 0.6, loss = "huber")

print("Start training the model_2")
# train the prediction model
model_2.fit(X_train, y_train) 
print("Model_2 trained")

# check the result
# we use mean absolute error, passing the data we used to train.
# model.predict runs the model
mae_train_2 = mean_absolute_error(y_train, model_2.predict(X_train))
print("Training Set MAE: %.2f" %mae_train_2)

# let's check with the test data
mae_test_2 = mean_absolute_error(y_test, model_2.predict(X_test))
print("Test Set MAE    : %.2f" %mae_test_2)

print("MAE training change: %.2f" %(mae_train_2 - mae_train))
print("MAE test change    : %.2f" %(mae_test_2 - mae_test))