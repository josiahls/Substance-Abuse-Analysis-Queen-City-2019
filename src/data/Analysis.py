import pandas as pd
import numpy as np

"""
We need to label each column based on these 3:

Fixed Variables    Variables that cannot be changed.
Dynamic Variables  Variables that can be changed.
Decision Variables Variables whose values we would like to minimize.

"""
# fdsafdsa
data = pd.read_csv('./traffic.csv')
columns = data.columns

columns_and_values = np.hstack((np.array(columns).reshape((-1, 1)), np.array([data[c].unique()[:] for c in columns]).reshape(-1, 1)))

print(columns_and_values)

IGNORED_VARIABLES = []
FIXED_VARIABLES = []
DYNAMIC_VARIABLES = []
DECISION_VARIABLES = []


""" Cleaning VARIABLES """
CATEGORICAL_VARIABLES = []
BOOLEAN_VARIABLES = []
HARD_NOT_NAN_VARIABLES = [] + DECISION_VARIABLES
MILT_TIME_VARIABLES = []









