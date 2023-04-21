# %% [markdown]
# Script splits the processed dataset into 80% training and 20% testing for normalisation by DataScalling script.

# %%
import pandas as pd                                                              # Data manipulation in dataframes
from sklearn.model_selection import(train_test_split)                            # Splits data frame into the training set and testing set

# %%
def Train_Test_Split(file):
    """      
    Input:      file             Pre-processed dataset

    Returns:    Training_Set     80% training set split
                Testing_Set      20% testing set split
                
    80% training and 20% testing split. Splits are shuffled randomly and index reset. Datasets are scaled by DatasetScalling script
    """
    AC_dataset                  = pd.read_csv(file, index_col = 0)  
        
    Training_Set, Testing_Set   = train_test_split(AC_dataset,train_size = 0.8)
        
    Training_Set.reset_index(drop=True, inplace = True)         #Drop index to avoid training on index values
    Testing_Set.reset_index(drop=True, inplace = True)          #Reset index after splitting for compatability with group fold CV
    
    Training_Set                = Training_Set.sample(frac = 1) #Shuffle data after splitting
    Testing_Set                 = Testing_Set.sample(frac = 1)
    
    Training_Set.to_csv('Training_Set.csv')
    Testing_Set.to_csv('Testing_Set.csv')

# %%
file = 'ScaledDataset.csv'
Train_Test_Split(file)


