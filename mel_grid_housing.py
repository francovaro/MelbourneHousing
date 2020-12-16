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
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(".\\Melbourne_housing_FULL.csv")

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

# Select and algorithm without defining hyperparameters
model = ensemble.GradientBoostingRegressor()

# define range for parameter
hyperparameters = { 'n_estimators': [200, 300],
    'learning_rate': [0.01, 0.02],
    'max_depth': [4, 6],
    'min_samples_split': [3, 4],
    'min_samples_leaf': [5, 6],
    'max_features': [0.8, 0.9],
    'loss': ['ls', 'lad', 'huber'],
    }

# defines the grid search
# n_jobs = nr of cpus
grid = GridSearchCV(model, hyperparameters, n_jobs = 4)

# train the model
grid.fit(X_train, y_train)

print("Best parameters")
print(grid.best_params_)

mae_train = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set MAE: %.2f" %mae_train)

mae_test = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set MAE    : %.2f" %mae_test)