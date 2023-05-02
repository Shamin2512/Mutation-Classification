# %% [markdown]
# 
# DatasetProcess.py peforms the pre-processing on the large dataset for SAAPpred.
# This includes removing NaNs, encoding class label, and simplifying the protein identifier to Uniprot Acession code.
# 
# Returns a combined, shuffled PDs and SNPs csv. 

# %%
import pandas as pd
import time
import random
from sklearn.preprocessing import LabelEncoder

# %%
def random_seed():
    """      
    Returns:    seed         Random seed from current time

    Generate a random seed to be used by all scripts
    """
    seed = random.randint(0, round(time.time()))
    random.seed(seed)
    
    return seed

# %%
def clean_data(PD_file, SNP_file, seed):
    """      
    Input:      PD_file          csv file of PD data
                SNP_file         csv file of SNP data

    Returns:    combined         Dataframe of combined SNP and PD data with error values removed

    Combine PD and SNP dataset, removes NaNs/ blank/ error spaces, encode dataset label and shuffle
    """
    
    df_pd = pd.read_csv(PD_file)
    df_snp = pd.read_csv(SNP_file)
    datasets = [df_pd, df_snp]
    df = pd.concat(datasets)

    #Remove NaNs/ blank/ error spaces, reset index to run from 0
    df.replace(to_replace=[' ', '?'], value = pd.NA, inplace=True)
    df.dropna(inplace = True)

    #Shuffle data to remove patterns and reset index
    df = df.sample(frac = 1, random_state= seed)
    df.reset_index(drop=True, inplace = True)

    #Encodes class labels to numeric values (0 or 1)
    df['dataset'] = 1 - (LabelEncoder().fit_transform(df['dataset']))   # Subtract from 1 so that PD = 1 and SNP = 0
    combined = df
        
    return combined

# %%
def identifer(combined):
    """      
    Input:      combined         Dataframe of pre-processed, combined SNP and PD data

    Returns:    cleaned          Dataframe of mutation data with UniProt Acession Code identifier

    Simplify the protein identifier column to UniProt Acession Code
    """

    AC_codes = combined.iloc[:, 0].str.extract(r':(\w+):')
    
    combined.drop(['num:uniprotac:res:nat:mut:pdbcode:chain:resnum:mutation:structuretype:resolution:rfactor'], axis=1, inplace=True) #Remove original column header
    combined.insert(0, 'AC Code', AC_codes)
    cleaned = combined

    return cleaned

# %%
def distance_feature(cleaned):
    """      
    Input:      cleaned                 Dataframe of pre-processed SNP and PD data

    Returns:    Dataset_Feature         Dataframe with additional SprotFTdist features
                Dataset_NoFeature       Dataframe without additional SprotFTdist features

    Output the pre-processed data as csv files
    """
    Dataset_Feature   = cleaned 
    Dataset_NoFeature = cleaned.drop(['SprotFTdist-ACT_SITE','SprotFTdist-BINDING','SprotFTdist-CA_BIND','SprotFTdist-DNA_BIND','SprotFTdist-NP_BIND','SprotFTdist-METAL','SprotFTdist-MOD_RES','SprotFTdist-CARBOHYD','SprotFTdist-MOTIF','SprotFTdist-LIPID'], axis = 1, inplace = False)
    
    Dataset_Feature.to_csv('Dataset_Feature.csv')
    Dataset_NoFeature.to_csv('Dataset_NoFeature.csv')

# %%
""" Main program """

seed = random_seed()
with open("seed.txt", "w") as f:        # write seed to text to be used by other scripts
    f.write(str(seed))
    
PD_file = input("Enter PD file: ")
SNP_file = input("Enter SNP file: ")

start = time.time()
combined = clean_data(PD_file, SNP_file, seed)
cleaned = identifer(combined)
distance_feature(cleaned)

end = time.time()


