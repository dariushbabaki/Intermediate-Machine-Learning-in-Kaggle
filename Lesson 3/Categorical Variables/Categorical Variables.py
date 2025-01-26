# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", '../input/train.csv')
    os.symlink("../input/home-data-for-ml-course/test.csv", '../input/test.csv')
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex3 import *
print("Setup Complete")

# Load the training and validation sets
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Print the first five rows of the data
X_train.head()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to compare different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Step 1: Drop columns with categorical data
# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# Check your answers
step_1.check()

# Lines below will give you a hint or solution code
#step_1.hint() 
#step_1.solution()

# Print MAE for this approach
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Investigate the 'Condition2' column
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

# Check your answer (Run this code cell to receive credit!)
step_2.a.check()

#step_2.a.hint

# Step 2: Ordinal encoding
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train [col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if
set (X_valid[col]).issubset (set (X_train [col]))]

# Problematic columns that will be dropped from the dataset bad_label_cols = list(set(object_cols) -set (good_label_cols))

print('Categorical columns that will be ordinal encoded:, good_label_cols) print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

# Part B: Ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

# Drop problematic categorical columns
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[good_label_cols] = encoder.fit_transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = encoder.transform(label_X_valid[good_label_cols])

# Check your answer
step_2.b.check()

# Lines below will give you a hint or solution code
#step_2.b.hint()
#step 2.b.solution()

# Print MAE for this approach
print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Get number of unique entries in each column with categorical data
object_nunique = list(map (lambda col: X_train[col].nunique(), object_cols)) 
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key-lambda x: x[1])

# Step 3: Investigating cardinality
# Part A
# Fill in the line below: How many categorical variables in the training data # have cardinality greater than 10?
high_cardinality_numcols = sum(1 for col in object_cols if X_train[col].nunique() > 10)

# Fill in the line below: How many columns are needed to one-hot encode the #Neighborhood' variable in the training data?
num_cols_neighborhood = X_train['Neighborhood'].nunique()

# Check your answers
step_3.a.check()

# Lines below will give you a hint or solution code
#step 3.a.hint()
#step 3.a. solution()

# Part B
# Fill in the line below: How many entries are added to the dataset by
# replacing the column with a one-hot encoding?
OH_entries_added = 990000

# Fill in the line below: How many entries are added to the dataset by # replacing the column with an ordinal encoding?
label_entries_added = 0

# Check your answers
step_3.b.check()

# Lines below will give you a hint or solution code
#step_3.b.hint()
#step_3.b.solution()

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

# Step 4: One-hot encoding
from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

OH_cols_train = pd.DataFrame(onehot_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(onehot_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_train.columns = onehot_encoder.get_feature_names_out(low_cardinality_cols)
OH_cols_valid.columns = onehot_encoder.get_feature_names_out(low_cardinality_cols)

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

OH_X_train = pd.concat([X_train.drop(object_cols, axis=1), OH_cols_train], axis=1)
OH_X_valid = pd.concat([X_valid.drop(object_cols, axis=1), OH_cols_valid], axis=1)

# Check your answer
step_4.check()

#Lines below will give you a hint or solution code
#step_4.hint()
#step 4.solution()

# Print MAE for this approach
print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# (Optional) Your code here
