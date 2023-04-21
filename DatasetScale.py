# %% [markdown]
# This script peforms the scaling of data for SAAPpred.
# Min-Max scaling scales all continuous (non-categorical) features to be between 0-1

# %%
""" Imports the required libraries and packages """

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

# %%
def open_dataset(file):
    """      
    Input:      file              CSV file to scale 

    Returns:    data              Dataframe of data to scale
            
    Open the file for scaling
    """    
    data = pd.read_csv(file, index_col=0)
    data.reset_index(drop = True, inplace = True)
    
    return data

# %%
def columns(data):
    """      
    Input:      data              Dataframe of data to scale

    Returns:    bool_col          List of columns to not scale
                scale_col         List of columns to scale
            
    Identifies the continuous data columns to scale
    """    
    all_col = data.columns.to_list()                             #List of all columns
    
    bool_col = [] 
    for col in data.columns:                                     # Boolean and pre-scaled columns that will not be scaled
        if data[col].min() == 0 and data[col].max() == 1:
            bool_col.append(col) 
        elif data[col].nunique() == 0 or data[col].nunique() == 1:
            bool_col.append(col) 
            
    bool_col.insert(0,'AC Code')                                 # Adds identifier

    scale_col = [col for col in all_col if col not in bool_col]
    
    return all_col, bool_col, scale_col

# %%
# max_value = np.max(train[scale_col])
# test_capped = np.clip(test[scale_col], 0, max_value)

# %%
def scale(data, all_col, bool_col, scale_col):
    """      
    Input:      bool_col          List of columns to not scale
                scale_col         List of columns to scale

    Returns:    scaled            Dataset scaled 0-1
            
    Scales the data using Min Max and returns it as a dataframe
    """    
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data.drop(bool_col, axis = 1))
    scaled = pd.DataFrame(scaled, columns = scale_col)
    scaled = pd.concat([data[bool_col], scaled],axis=1)
    
    scaled = scaled[all_col]
    
    return scaled

# %%
def output_csv(scaled):
    """      
    Input:      scaled            Dataset scaled 0-1
            
    Exports scaled dataframe as .csv
    """ 
    scaled.to_csv("ScaledDataset.csv")  

# %%
file = input("Enter file for scaling: ")
# file = 'AC_dataset.csv'
data = open_dataset(file)
all_col, bool_col, scale_col = columns(data)
scaled = scale(data, all_col, bool_col, scale_col)
output_csv(scaled)


