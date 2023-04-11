# %%
"""
This script peforms the pre-processing on the large dataset for SAAPpred.
This includes removing NaNs, encoding class label, and converting protein identifier to Uniprot Acession code.

Returns a combined, shuffled dataset of PDs and SNPs. 
"""
from urllib import request
import sys
import re
import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# %%
def clean_data(PD_file, SNP_file):
    df_pd = pd.read_csv(PD_file)
    df_snp = pd.read_csv(SNP_file)
    datasets = [df_pd, df_snp]
    df = pd.concat(datasets)

    #Remove NaNs/ blank/ error spaces, reset index to run from 0
    df.replace(to_replace=[' ', '?'], value = pd.NA, inplace=True)
    df.dropna(inplace = True)

    #Shuffle data to remove patterns and reset index
    df = df.sample(frac = 1)
    df.reset_index(drop=True, inplace = True)

    #Encodes class labels to numeric values (0 or 1)
    df['dataset'] = 1 - (LabelEncoder().fit_transform(df['dataset']))   # Subtract from 1 so that PD = 1 and SNP = 0
    cleaned = df
        
    return cleaned

# %%
def identifer(cleaned):

    AC_codes = cleaned.iloc[:, 0].str.extract(r':(\w+):')
    
    cleaned.drop(['num:uniprotac:res:nat:mut:pdbcode:chain:resnum:mutation:structuretype:resolution:rfactor'], axis=1, inplace=True) #Remove original column header
    cleaned.insert(0, 'AC Code', AC_codes)

    return cleaned, AC_codes

# %%
def distance_feature(cleaned):
    AC_dataset_Nofeature = cleaned.drop(['SprotFTdist-ACT_SITE','SprotFTdist-BINDING','SprotFTdist-CA_BIND','SprotFTdist-DNA_BIND','SprotFTdist-NP_BIND','SprotFTdist-METAL','SprotFTdist-MOD_RES','SprotFTdist-CARBOHYD','SprotFTdist-MOTIF','SprotFTdist-LIPID'], axis = 1, inplace = False)
    AC_dataset_feature = cleaned
    
    return(AC_dataset_feature, AC_dataset_Nofeature)

# %%
""" Main program """
start = time.time()
print("Running script...")

PD_file = "pd.csv"  
SNP_file = "snp.csv" 
cleaned = clean_data(PD_file, SNP_file)
cleaned, AC_Codes = identifer(cleaned)
AC_dataset_feature, AC_dataset_Nofeature = distance_feature(cleaned)

AC_dataset_feature.to_csv('Dataset_Feature.csv')
AC_dataset_Nofeature.to_csv('Dataset_NoFeature.csv')

end = time.time()


