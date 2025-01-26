#Setup
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
      os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv") 
      os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex2 import * 
print("Setup Complete")

#Load Data
import pandas as pd

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id') 
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors 
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True) 
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train.head()

#Investigate Missing Values (Step 1)
# Shape of training data (num_rows, num_columns) 
print(X_train.shape)

# Number of missing values in each column of training data 
missing_val_count_by_column = X_train.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column > 0])

 # Fill in the line below: How many rows are in the training data?
num_rows = X_train.shape[0]

# Fill in the line below: How many columns in the training data 
# have missing values?
num_cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].shape[0]

# Fill in the line below: How many missing entries are contained in # all of the training data?
tot_missing = missing_val_count_by_column.sum()

# Check your answers
step_1.a.check()

# Lines below will give you a hint or solution code
#step_1.a.hint() 
#step_1.a.solution()

# Check your answer (Run this code cell to receive credit!) 
step 1.b.check()

#step_1.b.hint()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset (X_train, X_valid, y_train, y_valid):
       model = RandomForestRegressor (n_estimators=100, random_state=0) 
       model.fit(X_train, y_train)
       preds = model.predict(X_valid)
       return mean absolute_error(y_valid, preds)
       
#Drop Columns with Missing Values (Step 2)
# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data 
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Check your answers
step_2.check()

# Lines below will give you a hint or solution code
#step_2.a.hint() 
#step_2.a.solution()

print("MAE (Drop columns with missing values):")
print(score_dataset (reduced_X_train, reduced_X_valid, y_train, y_valid))

#Impute Missing Values (Step 3)
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
imputer = SimpleImputer(strategy='mean')
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Check your answers
step_3.a.check()

# Lines below will give you a hint or solution code
#step_3.a.hint() 
#step_3.a.solution()

print("MAE (Imputation):")
print(score_dataset (imputed_X_train, imputed_X_valid, y_train, y_valid))

# Check your answer (Run this code cell to receive credit!) 
step_3.b.check()

#step_3.b.hint()

#Preprocess Test Data (Step 4)
# Preprocessed training and validation features
from sklearn.impute import SimpleImputer

# Imputation for missing values
imputer = SimpleImputer(strategy='mean')  # Your code here
final_X_train = pd.DataFrame(imputer.fit_transform(X_train))  # Your code here
final_X_valid = pd.DataFrame(imputer.transform(X_valid))  # Your code here

# Put back the column names
final_X_train.columns = X_train.columns  # Your code here
final_X_valid.columns = X_valid.columns  # Your code here

# Check your answers
step_4.a.check()

# Lines below will give you a hint or solution code
# step_4.a.hint()
# step_4.a.solution()

# Define and fit model
model = RandomForestRegressor (n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print (mean absolute_error (y_valid, preds_valid))

# Fill in the line below: preprocess test data 
final_X_test = pd.DataFrame(imputer.transform(X_test))
final_X_test.columns = X_test.columns 

# Fill in the line below: get test predictions 
preds_test = model.predict(final_X_test)

# Check your answers
step_4.b.check()

# Lines below will give you a hint or solution code
# step_4.b.hint()
# step_4.b.solution()

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
