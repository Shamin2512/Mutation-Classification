# %% [markdown]
# SAAPpred script that predicts protein pathogenicty from SAAPdap data, using XGBoost. 
# Goal is to predict SNP or PD with MCC > 0.7.
#         
# All CV models are directly used for final testing. Default params

# %% [markdown]
# ### Import library

# %%
""" Imports the required libraries and packages """

import pandas as pd                                                              # Data manipulation in dataframes
import numpy as np                                                               # Array manipulation
import xgboost as xgb                                                            # Gradient boosting package
import pickle                                                                    # Saving/loading GBM files
import hyperopt

import random as rd                                                              # Random seed generation
import time                                                                      # Time program run time
import sys

from xgboost import plot_importance

from sklearn.metrics import(
    matthews_corrcoef,                                                           # MCC for evaluation
    confusion_matrix,                                                            # Confusion matrix for classification evalutation
    )

from sklearn.model_selection import(
    GroupKFold                                                                       # K-fold CV 
        )

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK                            # Functions for minimising cost functions
from hyperopt.pyll.base import scope
from functools import partial

np.set_printoptions(precision = 3,threshold=np.inf, suppress=True)               # Full array printing

# %% [markdown]
# #### Open training and testing data

# %%
def open_data(file_train, file_test):
    """      
    Returns:    Training_Set     Normalised 80% training set split
                Testing_Set      Normalised 20% testing set split
                seed             Pre-generated seed
            
    Open the normalised training and testing data, and the generated seed
    """
    Training_Set = pd.read_csv(file_train, index_col = 0)
    Testing_Set = pd.read_csv(file_test,index_col = 0)
    
    with open("seed.txt", "r") as f:
        seed = int(f.read().strip())
    
    return Training_Set, Testing_Set, seed

# %%
def learning_data(Training_Set, Testing_Set):
    """      
    Input:      Training_Set     Scaled 80% training set split
                Testing_Set      20% testing set split

    Returns:    TrainData        Training features 
                TrainLabels      Training labels
                TestData         Testing features 
                TestLabels       Testing labels
            
    Separates training and testing data into features and labels
    """
    TrainData     = Training_Set.drop(['AC Code','dataset'], axis =1)  
    TrainLabels   = Training_Set['dataset']
    
    TestData     = Testing_Set.drop(['AC Code','dataset'], axis =1)  
    TestLabels   = Testing_Set['dataset']        
    
    return (TrainData, TrainLabels, TestData, TestLabels)

# %% [markdown]
# #### K Fold CV

# %%
def CV(Training_Set):
    """      
    Input:      Training_Set     80% training set split
            
    Returns:    IT_list         List of training features for each fold
                LT_list         List of training class labels for each fold
                IV_list         List of validation features for each fold
                LV_list         List of validation class labels for each fold

    K-fold CV with protein groups separated between training and validation sets for each fold. Creates 5 folds.
    """
    
    features     = Training_Set.drop(['dataset'], axis =1)         #Features for training
    labels       = Training_Set['dataset']                         #Class labels for training
    groups       = Training_Set['AC Code'].to_list()
    
    CV           = GroupKFold(n_splits = 5)                        #Creates 5 splits
    
    IT_list      = []
    LT_list      = []
    IV_list      = []
    LV_list      = []
    
    for train_idx, val_idx in CV.split(features, labels, groups):       #Generates the indices to be used for a training and validation split. Indicies are unique to train/ val sets

        Input_train                        = features.loc[train_idx]    #New dataframe from selected indices
        Classes_train                      = labels.loc[train_idx]
        Input_train.drop(['AC Code'], axis = 1, inplace = True)         #Group identifer not needed for training

                
        Input_val                          = features.loc[val_idx]
        Classes_val                        = labels.loc[val_idx]
        Input_val.drop(['AC Code'], axis   = 1, inplace = True)
        
        Input_train.reset_index(drop = True, inplace = True)            #Reset index of each set for compatability with balancing
        Classes_train.reset_index(drop = True, inplace = True)
        Input_val.reset_index(drop = True, inplace = True)
        Classes_val.reset_index(drop = True, inplace = True)

        IT_list.append(Input_train)       
        LT_list.append(Classes_train)
        IV_list.append(Input_val)
        LV_list.append(Classes_val)
    
    return(IT_list, LT_list, IV_list, LV_list)


# %% [markdown]
# #### Balancing

# %%
def find_minority_class(classData):
    """ 
    Input:        classData  Array of class labels

    Returns:      minClass   The label for the minority class
                  minSize    The number of items in the minority class
                  maxSize    The number of items in the majority class

    Find information about class size imbalance
    """
    
    Minority_count = 0
    Majority_count = 0
    for datum in classData:
        if datum == 1:
            Majority_count += 1
        elif datum == 0:
            Minority_count += 1

    minClass = 0
    minSize  = Minority_count
    maxSize  = Majority_count
    if Minority_count > Majority_count:
        minClass = 1
        minSize  = Majority_count
        maxSize  = Minority_count

    return minClass, minSize, maxSize

# %%
def balance(inData, classData, minClass, minSize):
    """ 
    Input:        inData          Dataframe of input data
                  classData       Series of classes assigned
                  minorityClass   class label for the minority class
                  minoritySize    size of the minority class

    Returns:      usedLines       array of indexes that are of interest for a balanced dataset

    Perform the actual balancing for a fold between SNPs and PDs
    """
    usedLines = [False] * len(inData)      #Array of false for length of data
    for i in range(len(inData)):
        if classData[i] == minClass:       #Balance directly with dataframe
            usedLines[i] = True            #True lines are SNP
            
    usedCount = 0
    while usedCount < minSize:
        i = rd.randrange(len(inData))
        if usedLines[i] == False:
            usedLines[i] = True
            usedCount += 1                 #Set PD lines "True", until equal to number of SNP lines

    return usedLines

# %%
def balance_data(inData, classData, usedLines):
    """     
    Input:      inData         Datframe of input training data
                classData      Series of classes assigned to training data
                usedLines      Array of line indexes to print

    Returns:    input_balance  Dataframe of balanced training features
                label_balance  Dataframe of balanced training labels
                       
    Create dataframe of the input training data and classes used. Index_list tracks the indicies between usedLines and inData, used to pull the required5 lines.
    """
    input_balance = []
    label_balance = []
    index_list = []
    
    for i in range(len(usedLines)):
        if usedLines[i] == True:
            index_list.append(i)
             
    input_balance = inData.iloc[index_list].reset_index(inplace = False, drop = True)
    label_balance = classData.iloc[index_list].reset_index(inplace = False, drop = True) 
    
    return input_balance, label_balance

# %%
def Balance_ratio(maxSize, minSize): 
    """ 
    Input:      maxSize     The number of items in the majority class
                minSize     The number of items in the minority class

    Returns:    BF          Number of balancing folds

    Calculate the number of balancing folds needed using ratio of majority to minority class size. Double to ensure sufficient
    majority class instances are sampled, then + 1 to make odd to allow weighted vote.
    """
    Divide = maxSize/minSize
    BF = ((2 * round(Divide)) + 1)    # ratio to nearest odd integer
    return BF

# %%
def Balance_Folds(BF, inData, classData, minClass, minSize):
    """ 
    Input:      BF                Number of balancing folds
                inData            Datframe of input training data
                classData         Series of classes assigned to training data
                minClass          The label for the minority class
                minSize           The number of items in the minority class
                                  
    Returns:    Input_folds       List of balanced training feature folds
                Output_folds      List of balanced training label folds

    Runs balance_data() for the number of balancing folds. Return lists of balanced folds features and labels
    where each item is the output of balance_data()
    """
    Input_folds  = []
    Output_folds = []

    for i in range(BF):
        usedLines                    = balance(inData, classData, minClass, minSize)
        input_balance, label_balance = balance_data(inData, classData, usedLines)
        
        Input_folds.append(input_balance)
        Output_folds.append(label_balance)
            
    return Input_folds, Output_folds

# %% [markdown]
# #### Training

# %%
def GBM_dmatrix(BF, Input_folds, Output_folds, ValData, Vallabel):
    """ 
    Input:      BF                Number of balancing folds
                Input_folds       List of balanced training feature folds
                Output_folds      List of balanced training label folds
                ValData           Validation features from CV fold
                ValLabel          Valiadation labels from CV fold
                                  
    Returns:    d_train_list      List of balanced training feature folds as DMatrix
                d_val             Validation data as Dmatrix

    Converts the balanced training data and validation data into Dmatrix for model training and evaluation
    """

    d_train_list =[]
    
    for i in range(BF):
        d_train = xgb.DMatrix(Input_folds[i], Output_folds[i])      #Create DMatrix for each training balanced fold
        d_train_list.append(d_train)
    d_val = xgb.DMatrix(ValData, Vallabel)

    return (d_train_list, d_val)


# %%
# def BF_fitting(BF, d_train_list, d_val, fold, pickle_file): 
#     """ 
#     Input:      BF                Number of balancing folds                      
#                 d_train_list      List of balanced training feature folds in DMatrix
#                 d_val             Validation data as Dmatrix
                
#     Returns:    BF_GBC            List of gradient boosted models trained on each balancing fold

#     Create model with the best hyperparameters, using output of Balance_Folds() as training data (as Dmatrix)
#     """     
#     BF_GBM = []
    
#     for fold_i in range(BF):
        
#         params = {
#             'booster': 'gbtree',
#             'tree_method': 'hist',
#             'objective': 'binary:logistic', 
#             'verbosity': 0,
#             'num_parallel_tree': 1,
#             'eval': 'error',
#             }
        
#         d_train = d_train_list[fold_i]                              #Dmatrix for each balanced fold
#         model = xgb.train(params,                                   #Generates and fits a GBC for each training balanced fold
#                           d_train,
#                           num_boost_round = 1500,
#                         #   evals = [(d_train, 'train'), (d_val, 'validation')],
#                         #   early_stopping_rounds = 20,
#                           )
        
#         filename = f"CV_{fold + 1}_model_{fold_i + 1}.pkl"
#         with open(filename, "wb") as f:
#             pickle.dump(model, f)
            
#         BF_GBM.append(model)
#         pickle_file.append(filename)
        
#     return BF_GBM, pickle_file

# %%
def BF_fitting(inData, classData, ValData, Vallabel, fold, pickle_file): 
    """ 
    Input:      BF                Number of balancing folds                      
                d_train_list      List of balanced training feature folds in DMatrix
                d_val             Validation data as Dmatrix
                
    Returns:    BF_GBC            List of gradient boosted models trained on each balancing fold

    Create model with the best hyperparameters, using output of Balance_Folds() as training data (as Dmatrix)
    """     
    BF_GBM = []     
    params = {
        'booster': 'gbtree',
        'tree_method': 'hist',
        'objective': 'binary:logistic', 
        'verbosity': 0,
        'num_parallel_tree': 1,
        }
    
    d_train = xgb.DMatrix(inData, classData)
    d_val   = xgb.DMatrix(ValData, Vallabel)
    model = xgb.train(params,                                   #Generates and fits a GBC for each training balanced fold
                        d_train,
                        num_boost_round = 1500,
                    #   evals = [(d_train, 'train'), (d_val, 'validation')],
                    #   early_stopping_rounds = 20,
                        )
    
    filename = f"CV_{fold + 1}_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
        
    BF_GBM.append(model)
    pickle_file.append(filename)
        
    return BF_GBM, pickle_file, d_val

# %% [markdown]
# #### Validation

# %%
def BF_predict(BF_GBM, d_val):
    """ 
    Input:      BF_GBM            List of GBMs trained on balancing folds
                d_val             Validation data as Dmatrix

                
    Returns:    Prob_matrix     List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the validation set.
    """
    
    Prob_matrix = []
    for i in range(len(BF_GBM)):
        Prob = BF_GBM[i].predict(d_val)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
        Prob_matrix.append(Prob)   
        
    return Prob_matrix

# %%
def Weighted_Vote(Prob_matrix):
    """ 
    Input:      Prob_matrix     List of arrays. 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability

    Returns:    Final_vote      Weighted vote classification

    Calculate the final weighted vote using confidence scores (Sc) from Prob_matrix. Binary classification formula for:
    Predictor states its prediction/ confidence scores are between 0.0 and 1.0 for each class
    """
    PD_prob_matrix = Prob_matrix 
    
    SNP_prob_matrix = []
    for i in range(len(Prob_matrix)):               #SNP probabilites are 1 - (PD probabilites)
        sub = 1 - Prob_matrix[i]
        SNP_prob_matrix.append(sub)
            
    Sum_SNP = np.sum(SNP_prob_matrix, axis = 0)     #Sum of all SNP confidence scores. 1D Array
    Sum_PD  = np.sum(PD_prob_matrix, axis = 0)      #Sum of all PD confidence scores. 1D Array
                                                    
    Vote_arr  = [] 

    for i in range(len(Sum_PD)):
        if Sum_PD[i] >= Sum_SNP[i]:
            Vote_arr.append([1])                    #Append PD classifications to list
        elif Sum_SNP[i] > Sum_PD[i]:
            Vote_arr.append([0])                    #Append SNP classifications to list

    Final_vote = np.stack(Vote_arr)                 #Converts list of arrays to a 2D array
    Final_vote = Final_vote.ravel()                 #Flattens 2D array to 1D array

    return(Final_vote, Sum_PD, Sum_SNP)             #Returns the final confidence scores


# %%
def CV_evaluation(d_val, Final_vote):
    """ 
    Input:      d_val             Validation data as Dmatrix
                Final_vote        Weighted vote classification
                
    Evaluates a CV fold's trained model with MCC
    """
    Output_pred = Final_vote
    TrueLabel   = d_val.get_label()
        
    CV_MCC = matthews_corrcoef(TrueLabel, Output_pred)

    return CV_MCC

# %% [markdown]
# ### Test on testing set

# %%
def final_BF_predict(pickle_file, TestData):
    """ 
    Input:      BF_RFC            List of RFCs trained on balancing folds
                d_test            Testing data as Dmatrix

                
    Returns:    Prob_matrix     List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the testing set.
    """
    all_prob_matrix = []
    all_models = []
    prefix = f"CV_"
    d_test = xgb.DMatrix(TestData)
    
    for file in pickle_file:
        if file.startswith(prefix):
            with open(file, "rb") as f:
                    model = pickle.load(f)
                    Prob = model.predict(d_test)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
                    all_prob_matrix.append(Prob)            
            
        
    return all_prob_matrix

# %%
def final_evaluation(Prob_matrix, TestLabels):
    """ 
    Input:      all_prob_matrix    List of all predicted probabilites from all optimised models
                TestLabels         True labels from unseen 20% testing data

    Returns:    MCC_final          Final MCC evaluation

    Calculate the final predictions with weighted vote using confidence scores. 
    Evaluate votes agains true labels to give the final MCC
    """
        
    PD_prob_matrix = Prob_matrix

    SNP_prob_matrix = []
    for i in range(len(PD_prob_matrix)):                 #SNP probabilites are 1 - (PD probabilites)
        sub = 1 - PD_prob_matrix[i]
        SNP_prob_matrix.append(sub)
            
    Sum_SNP = np.sum(SNP_prob_matrix, axis = 0)     #Sum of all SNP confidence scores. 1D Array
    Sum_PD  = np.sum(PD_prob_matrix, axis = 0)      #Sum of all PD confidence scores. 1D Array
                                                    
    Vote_arr  = [] 

    for i in range(len(Sum_PD)):
        if Sum_PD[i] >= Sum_SNP[i]:
            Vote_arr.append([1])                    #Append PD classifications to list
        elif Sum_SNP[i] > Sum_PD[i]:
            Vote_arr.append([0])                    #Append SNP classifications to list

    Final_vote = np.stack(Vote_arr)                 #Converts list of arrays to a 2D array
    Final_vote = Final_vote.ravel()                 #Flattens 2D array to 1D array
    
    MCC_final = matthews_corrcoef(TestLabels, Final_vote)
        
    return(MCC_final)


# %% [markdown]
# ### Main Program

# %%
start = time.time()

# file_train                = input("Enter file for training: ")
# file_test                 = input("Enter file for testing: ")
file_train                  = "STraining_Set.csv"
file_test                   = "STesting_Set.csv"
Training_Set, Testing_Set, seed   = open_data(file_train, file_test)
rd.seed(seed)

Score_list = []

# for i in range(0,15):
pickle_file = []
CV_score = []

print("Opening dataset...")
TrainData, TrainLabels, TestData, TestLabels = learning_data(Training_Set, Testing_Set)   

print("Performing Group fold cross validation...")           
IT_list, LT_list, IV_list, LV_list = CV(Training_Set)     

for fold in range(len(IT_list)):          
    inData                      = IT_list[fold]
    classData                   = LT_list[fold]
    ValData                     = IV_list[fold]
    Vallabel                    = LV_list[fold]

    # print(f"[Fold {fold + 1}] Balancing...")
    # minClass, minSize, maxSize  = find_minority_class(classData)   
    # BF                          = Balance_ratio(maxSize, minSize)                        
    # Input_folds, Output_folds   = Balance_Folds(BF, inData, classData, minClass, minSize)
    # d_train_list, d_val         = GBM_dmatrix(BF, Input_folds, Output_folds, ValData, Vallabel)
    
    print(f"[Fold {fold + 1}] CV Training...")
    
    # BF_GBC, pickle_file         = BF_fitting(BF, d_train_list, d_val, fold, pickle_file)
    BF_GBC, pickle_file, d_val           = BF_fitting(inData, classData, ValData, Vallabel, fold, pickle_file)


    Prob_matrix                 = BF_predict(BF_GBC, d_val)
    Final_vote, Sum_PD, Sum_SNP = Weighted_Vote(Prob_matrix)
    CV_MCC                      = CV_evaluation(d_val, Final_vote)  
    CV_score.append(CV_MCC)
              
    print(f"Fold {fold + 1} MCC:\n{CV_MCC}\n")

best_fold = (CV_score.index(max(CV_score)) + 1)

#Testing
print("Testing...") 
Prob_matrix = final_BF_predict(pickle_file, TestData)
MCC_final = final_evaluation(Prob_matrix, TestLabels) 
# print(MCC_final) 

Score_list.append(MCC_final)
#loop     
end = time.time()
# plot(Score_list)
print(f"Final evaluation:{np.mean(Score_list)} \u00B1 {np.std(Score_list)}\n\nLowest score:{min(Score_list)}\nHighest score:{max(Score_list)}\n\nRun time: {end-start}")


