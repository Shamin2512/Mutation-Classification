# %% [markdown]
# DatasetScale.py peforms the scaling of ll non-boolean features.
# 
# Returns min-max and standard scaling scaled csv

# %%
""" Imports the required libraries and packages """

import pandas as pd
import numpy as np
from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler
                                   ) 

# %%
def open_dataset(file):
    """      
    Input:      file              csv file to scale 

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
            
    Identify the continuous data columns to scale
    """    
    all_col = data.columns.to_list()                             #List of all columns
    
    bool_col = [] 
    for col in data.columns:                                     # Boolean and pre-scaled columns that will not be scaled
        if data[col].min() == 0 and data[col].max() == 1:
            bool_col.append(col) 
        elif data[col].nunique() == 0 or data[col].nunique() == 1:
            bool_col.append(col) 
            
    bool_col.insert(0,'AC Code')                                 # Adds identifier

    scale_col = [col for col in all_col if col not in bool_col]  # Columns to be scaled
    
    return all_col, bool_col, scale_col

# %%
def min_max_scale(data, all_col, bool_col, scale_col):
    """      
    Input:      bool_col          List of columns to not scale
                scale_col         List of columns to scale

    Returns:    scaled            Dataset normalised 0-1
            
    Normalise the data using Min Max and return it as a dataframe
    """    
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data.drop(bool_col, axis = 1))
    scaled = pd.DataFrame(scaled, columns = scale_col)
    scaled = pd.concat([data[bool_col], scaled],axis=1)
    
    min_max_scaled = scaled[all_col]
    
    return min_max_scaled

# %%
def standard_scale(data, all_col, bool_col, scale_col):
    """      
    Input:      bool_col          List of columns to not scale
                scale_col         List of columns to scale

    Returns:    scaled            Dataset scaled 0-1
            
    Standardise the data using standard scaling and return it as a dataframe
    """    
    scaler = StandardScaler()

    scaled = scaler.fit_transform(data.drop(bool_col, axis = 1))
    scaled = pd.DataFrame(scaled, columns = scale_col)
    scaled = pd.concat([data[bool_col], scaled],axis=1)
    
    standard_scaled = scaled[all_col]
    
    return standard_scaled

# %%
def output_csv(min_max_scaled, standard_scaled):
    """      
    Input:      scaled            Dataset scaled 0-1
            
    Exports scaled dataframes as .csv
    """ 
    min_max_scaled.to_csv("MMDataset.csv")
    standard_scaled.to_csv("SDataset.csv")    

# %%
"""Main program"""

file = input("Enter file for scaling: ")
data = open_dataset(file)
all_col, bool_col, scale_col = columns(data)
min_max_scaled = min_max_scale(data, all_col, bool_col, scale_col)
standard_scaled = standard_scale(data, all_col, bool_col, scale_col)

output_csv(min_max_scaled, standard_scaled)


