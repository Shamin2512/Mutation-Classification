#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This script peforms the pre-processing needed for MutationClassification program. 
This includes removing NaNs, encoding class label, and converting the PDB codes in a dataset to Uniprot accession codes.
Only converts the 4 alphanumeric characters, not the chain and residue number. 

Returns a processed dataset. 
"""

#!/usr/bin/env python3

from urllib import request
import sys
import re
import pandas as pd
import numpy as np
import time


# In[2]:


def clean_data(file):
    df = pd.read_csv('E2.csv')

    #Remove unrequired NaNs, blank spaces, reset index to run from 0
    df.dropna(inplace = True)
    df.replace(' ', '_', regex=True, inplace=True)
    df.reset_index(drop=True, inplace = True)

    #Encodes class labels to numeric values (0 or 1)
    cleaned_encoded = pd.get_dummies(df, columns=['dataset']) #Encode the PD and SNP columns
    cleaned = cleaned_encoded.drop(['dataset_snp'],axis = 1)
    
    return cleaned


# In[3]:


def group_data(cleaned):
    group_cleaned = cleaned.sort_values(by=['pdbcode:chain:resnum:mutation'])

    PDB_codes = []
    for i in range(len(group_cleaned)):
        PDB_codes.append(group_cleaned.iloc[i][0].partition(':')[0]) #Split the identifier and takes only PDB code

    group_cleaned.drop(['pdbcode:chain:resnum:mutation'], axis=1, inplace=True) #Remove 'pdbcode:chain:resnum:mutation' column
    group_cleaned.reset_index(inplace = True, drop = True)

    return group_cleaned, PDB_codes


# In[4]:


def PDBSWS(PDB_codes, group_cleaned):
    
    AC_codes = []
    for i in range(len(PDB_codes)):
        url = 'http://www.bioinf.org.uk/servers/pdbsws/query.cgi?plain=1&qtype=pdb' #REST output
        url += '&id=' + PDB_codes[i] #URL for the specific PDB code of interest
        
        result = request.urlopen(url).read() #Reads the URL
        result = str(result, encoding='utf-8') #Encodes the URL into utf-8 format
        result = result.replace('\n', '#') #Replaces all the new line returns with #, allowing easy pattern matches

        pattern  = re.compile('.*AC:\s+(.*?)#') #Recognises the accession code pattern
        match    = pattern.match(result) #Saves the pattern to match
        ac       = match.group(1) #Saves only the accession code to variable
        
        AC_codes.append(ac)
    group_cleaned.insert(0, 'AC Code', AC_codes)
    AC_dataset.to_csv('AC_dataset.csv)
                      
    return AC_codes, group_cleaned


# In[5]:


""" Main program """
start = time.time()

file = "E2.csv"
cleaned = clean_data(file)
group_cleaned, PDB_codes = group_data(cleaned)
AC_codes, AC_dataset = PDBSWS(PDB_codes, group_cleaned)
end = time.time()


# In[6]:


print(f"Time: {end-start} seconds\n\n{AC_dataset}")

