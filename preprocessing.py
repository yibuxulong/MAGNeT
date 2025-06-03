import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split


def split_train_test(data_condition, test_ratio=0.3, val_ratio=0.1):
    """
    Split the data into training and test sets
    """
    W_values = list(collections.Counter(data_condition[0]['W']).keys())
    V_values = list(collections.Counter(data_condition[0]['V']).keys())

    data_train, data_val, data_test = [], [], []

    for data_cur in data_condition:
        for w in W_values:
            for v in V_values:
                # Filter data by current W and V combination
                data_subset = data_cur[(data_cur['W'] == w) & (data_cur['V'] == v)]
                
                # Split into train, validation, and test sets
                train_data, temp_data = train_test_split(data_subset, test_size=test_ratio + val_ratio)
                val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio))
                
                data_train.append(train_data)
                data_val.append(val_data)
                data_test.append(test_data)

    data_train = pd.concat(data_train, ignore_index=True)
    data_val = pd.concat(data_val, ignore_index=True)
    data_test = pd.concat(data_test, ignore_index=True)
    
    return data_train, data_val, data_test
