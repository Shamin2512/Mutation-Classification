# %% [markdown]
# DatasetSplit.py script splits the scaled dataset into 80% training and 20% testing sets

# %%
import pandas as pd                                                              # Data manipulation in dataframes
from sklearn.model_selection import(train_test_split)                            # Splits data frame into the training set and testing set

# %%
def Train_Test_Split(file, seed):
    """      
    Input:      file             Pre-processed dataset

    Returns:    Training_Set     80% training set split
                Testing_Set      20% testing set split
                
    80% training and 20% testing split. Splits are shuffled randomly and index reset. Datasets are scaled by DatasetScalling script
    """
    AC_dataset                  = pd.read_csv(file, index_col = 0)  
        
    Training_Set, Testing_Set   = train_test_split(AC_dataset,train_size = 0.8, random_state= seed)
        
    Training_Set.reset_index(drop=True, inplace = True)         #Drop index to avoid training on index values
    Testing_Set.reset_index(drop=True, inplace = True)          #Reset index after splitting for compatability with group fold CV
    
    Training_Set                = Training_Set.sample(frac = 1, random_state = seed) #Shuffle data after splitting
    Testing_Set                 = Testing_Set.sample(frac = 1, random_state = seed)
    
    Training_Set.to_csv('Training_Set.csv')
    Testing_Set.to_csv('Testing_Set.csv')

# %%
"""Main program"""

with open("seed.txt", "r") as f:
    seed = int(f.read().strip())
    
file = input("Enter file for splitting: ")
Train_Test_Split(file, seed)


