# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
      os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv") 
      os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Print the first few rows of the training data
X_train.head()


from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
       model.fit(X_t, y_t)
       preds = model.predict(X_v)
       return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
      mae = score_model(models[i])
      print("Model %d MAE: %d" % (i+1, mae))

# Loop through the models to find the one with the lowest MAE
mae_values = []
for i in range(0, len(models)):
    mae = score_model(models[i])
    mae_values.append(mae)
    print(f"Model {i+1} MAE: {mae}")

# Find the model with the lowest MAE
best_model_index = mae_values.index(min(mae_values))  # Index of the best model
best_model = models[best_model_index]  # Assign the best model

# Check your answer
step_1.check()

# The lines below will show you a hint or the solution.
# step_1.hint()
# step_1.solution()

# Define a model
my_model = RandomForestRegressor(n_estimators=100, random_state=0)  # Adjust parameters based on the best model

# Check your answer
step_2.check()

# The lines below will show you a hint or the solution.
# step_2.hint()
# step_2.solution()

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
