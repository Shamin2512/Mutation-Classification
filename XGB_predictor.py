# %% [markdown]
# SAAPpred script that predicts protein pathogenicty from SAAPdap data, using XGBoost. 
# Goal is to predict SNP or PD with MCC > 0.7.
#         
# Uses all CV models for final testing

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

from sklearn.metrics import(
    matthews_corrcoef,                                                           # MCC for evaluation
    confusion_matrix,                                                            # Confusion matrix for classification evalutation
    )

from sklearn.model_selection import(
    GroupKFold                                                                   # K-fold CV with groups
        )

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK                            # Hyperopt search functions
from hyperopt.pyll.base import scope
from functools import partial                                                    # Function wrapping

np.set_printoptions(precision = 3,threshold=np.inf, suppress=True)               # Full array printing

# %% [markdown]
# ### Split dataset into training and validation sets

# %%
def open_data(file_train, file_test):
    """      
    Input:      file_train       Training_set csv
                file_test        Testing_set csv
                
    Returns:    Training_Set     Scaled 80% training set split
                Testing_Set      Scaled 20% testing set split
            
    Open the scaled training and testing data as dataframe
    """
    Training_Set = pd.read_csv(file_train, index_col = 0)
    Testing_Set = pd.read_csv(file_test,index_col = 0)
    
    return Training_Set, Testing_Set

# %%
def learning_data(Testing_Set):
    """      
    Input:      Testing_Set      Scalted 20% testing set split

    Returns:    TestData         Testing features 
                TestLabels       Testing labels
            
    Separates testing data into features and labels and create DMatrix
    """    
    TestData     = Testing_Set.drop(['AC Code','dataset'], axis =1)  
    TestLabels   = Testing_Set['dataset'] 
    d_test       = xgb.DMatrix(TestData, TestLabels)       
    
    return (TestData, TestLabels, d_test)

# %% [markdown]
# ## Outer Loop: Group Fold CV

# %%
def CV(Training_Set):
    """      
    Input:      Training_Set     80% training set split
            
    Returns:    IT_list         List of training features for each fold
                LT_list         List of training class labels for each fold
                IV_list         List of validation features for each fold
                LV_list         List of validation class labels for each fold

    5-fold-group CV with protein groups separated between training and validation sets for each fold
    """
    
    features     = Training_Set.drop(['dataset'], axis =1)         # Features for training
    labels       = Training_Set['dataset']                         # Class labels for training
    groups       = Training_Set['AC Code'].to_list()               # List of proteins for grouping
    
    CV           = GroupKFold(n_splits = 5)                        # Creates 5 splits
    
    IT_list      = []
    LT_list      = []
    IV_list      = []
    LV_list      = []
    
    for train_idx, val_idx in CV.split(features, labels, groups):       # Generates the indices to be used for a training and validation split. Indicies are unique to train/ val sets

        Input_train                        = features.loc[train_idx]    # ew dataframe from selected indices
        Classes_train                      = labels.loc[train_idx]
        Input_train.drop(['AC Code'], axis = 1, inplace = True)         # Group identifer not needed for training

        Input_val                          = features.loc[val_idx]
        Classes_val                        = labels.loc[val_idx]
        Input_val.drop(['AC Code'], axis   = 1, inplace = True)
        
        Input_train.reset_index(drop = True, inplace = True)            # Reset index of each set for compatability with balancing
        Classes_train.reset_index(drop = True, inplace = True)
        Input_val.reset_index(drop = True, inplace = True)
        Classes_val.reset_index(drop = True, inplace = True)

        IT_list.append(Input_train)       
        LT_list.append(Classes_train)
        IV_list.append(Input_val)
        LV_list.append(Classes_val)
    
    return(IT_list, LT_list, IV_list, LV_list)


# %% [markdown]
# ## Inner Loop:

# %% [markdown]
# ### Balancing

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
    usedLines = [False] * len(inData)      # Array of false for length of data
    for i in range(len(inData)):
        if classData[i] == minClass:       # Balance directly with dataframe
            usedLines[i] = True            # True lines are SNP
            
    usedCount = 0
    while usedCount < minSize:
        i = rd.randrange(len(inData))
        if usedLines[i] == False:
            usedLines[i] = True
            usedCount += 1                 # Set PD lines "True", until equal to number of SNP lines

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
    BF = (2 * round(Divide)) + 1    # Double ratio to nearest integer
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

    Run balance_data() for the number of balancing folds. Return lists of balanced folds features and labels
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
# ### Training

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

    Convert the balanced training data and validation data into DMatrix for XGB model training and evaluation
    """

    d_train_list =[]
    
    for i in range(BF):
        d_train = xgb.DMatrix(Input_folds[i], Output_folds[i])      # Create DMatrix for each training balanced fold
        d_train_list.append(d_train)
    d_val = xgb.DMatrix(ValData, Vallabel)

    return (d_train_list, d_val)


# %%
def MCC_eval_metric(pred, d_val):
    """ 
    Input:      pred              Prediction from a boosted tree during training
                d_val             Validation data as Dmatrix
    
    Returns:    mcc               The MCC from a boosted tree round

    MCC as a custom evaluation metric for early stopping
    """
    true_label = d_val.get_label()   
    pred_label = np.round(pred) 
    
    return 'mcc', matthews_corrcoef(true_label, pred_label )

# %%
def hyperopt_space():
    """ 
    Returns:   Space         Parameter space for hyperparameter searching

    Define paramater space for Hyperopt to search throug
   """  
    space = {
        'num_boost_round': scope.int(hp.quniform('num_boost_round', 1250, 1750, 250)),
        'max_depth': scope.int(hp.quniform('max_depth', 9, 13, 1)),
        'eta': hp.uniform('eta', 0.45, 0.75),
        }
     
    return space


# %%
def objective(params, d_train_list, d_val, MCC_eval_metric): 
    """ 
    Input:      params            Search paramaters parsed in fmin function()                     
                d_train_list      List of balanced training feature folds in DMatrix
                d_val             Validation data as Dmatrix
                
    Returns:    loss, status      The MCC from evaluating hyperparameters during search

    Define the model that Hyperopt will optimise hyperparameters for
    """     
    max_depth       = params['max_depth']
    eta             = params['eta']
    num_boost_round = params['num_boost_round']
    
    settings = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'disable_default_eval_metric': 1,
        'verbosity': 1,
        'max_depth': max_depth,
        'eta': eta,
            } 
                    
    model = xgb.train(settings,
                      d_train_list,
                      num_boost_round = num_boost_round,
                      evals = [(d_val, 'Model')],
                      verbose_eval = False,
                      early_stopping_rounds = 40,
                      custom_metric = MCC_eval_metric,
                        )
    # Generates and fits a GBM for each training balanced fold
    
    pred = model.predict(d_val)
    MCC = matthews_corrcoef(d_val.get_label(), pred.round())
                                                                         
    return {'loss': -MCC, 'status': STATUS_OK}


# %%
def BF_fitting(BF, d_train_list, fold, best_param, pickle_file): 
    """ 
    Input:      BF                Number of balancing folds                      
                d_train_list      List of balanced training feature folds in DMatrix
                d_val             Validation data as Dmatrix
                
    Returns:    BF_GBC            List of gradient boosted models trained on each balancing fold

    Create model with the best hyperparameters, using output of Balance_Folds() as training data (as Dmatrix)
    """     
    BF_GBM = []
    
    for fold_i in range(BF):
        
        params = {
            'booster': 'gbtree',
            'tree_method': 'hist',
            'objective': 'binary:logistic', 
            'disable_default_eval_metric': 0,
            'verbosity': 0,
            'num_parallel_tree': 1,
            'max_depth': int(best_param[fold]['max_depth']),
            'eta': best_param[fold]['eta'],
            }
        
        d_train = d_train_list[fold_i]                              # Dmatrix for each balanced fold
        model = xgb.train(params,                                   
                          d_train, 
                          num_boost_round = int(best_param[fold]['num_boost_round']),
                          )
        # Generates and fits a GBC for each training balanced fold
        
        filename = f"CV_{fold + 1}_model_{fold_i + 1}.pkl" 
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        # Save each model using pickle
        
        BF_GBM.append(model)
        pickle_file.append(filename)
        
    return BF_GBM, pickle_file

# %% [markdown]
# ### Validation

# %%
def BF_predict(BF_GBM, d_val):
    """ 
    Input:      BF_RFC            List of RFCs trained on balancing folds
                d_val             Validation data as Dmatrix

    Returns:    Prob_matrix     List ofd arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability
    
    Use trained model to predict probability of every datapoint in validation set
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

    return(Final_vote)                              #Returns the final confidence scores


# %%
def CV_evaluation(d_val, Final_vote):
    """ 
    Input:      d_val             Validation data as Dmatrix
                Final_vote        Weighted vote classification
                
    Evaluates CV fold with MCC
    """
    Output_pred = Final_vote
    TrueLabel   = d_val.get_label()
        
    CV_MCC = matthews_corrcoef(TrueLabel, Output_pred)

    return CV_MCC

# %% [markdown]
# ## Final evaluation on testing set 

# %%
def final_BF_predict(pickle_file, d_test):
    """ 
    Input:      BF_RFC            List of RFCs trained on balancing folds
                d_test            Testing data as Dmatrix

                
    Returns:    all_prob_matrix   List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                  2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the testing set.
    """
    all_prob_matrix = []
    prefix = f"CV_"
        
    for file in pickle_file:
        if file.startswith(prefix):
            with open(file, "rb") as f:
                    model = pickle.load(f)
                    prob = model.predict(d_test)     #Predicts the probability of an instance belonging to PD (label = 1) and SNP (label = 0)
                    all_prob_matrix.append(prob)
                    
    return all_prob_matrix

# %% [markdown]
# ### Main Program

# %%
start = time.time()

pickle_file = []
CV_score = []
best_param = []

file_train = input("Enter file for training: ")
file_test = input("Enter file for testing: ")

print("Opening dataset...")
Training_Set, Testing_Set    = open_data(file_train, file_test)
TestData, TestLabels, d_test = learning_data(Testing_Set)  

print("Performing 5-fold group CV...")            
IT_list, LT_list, IV_list, LV_list = CV(Training_Set)     

for fold in range(len(IT_list)):          
    inData = IT_list[fold]
    classData = LT_list[fold]
    ValData = IV_list[fold]
    Vallabel = LV_list[fold]
    
    print(f"[Fold {fold + 1}] Balancing...")
    minClass, minSize, maxSize  = find_minority_class(classData)   
    BF                          = Balance_ratio(maxSize, minSize)                        
    Input_folds, Output_folds   = Balance_Folds(BF, inData, classData, minClass, minSize)
    d_train_list, d_val         = GBM_dmatrix(BF, Input_folds, Output_folds, ValData, Vallabel)
    
    print(f"[Fold {fold + 1}] Hyperparameter searching...")
    space = hyperopt_space()
    trials = Trials()
    fmin_objective = partial(objective,
                                d_train_list = d_train_list[rd.randrange(BF)],
                                d_val = d_val,
                                MCC_eval_metric = MCC_eval_metric)
    
    best = fmin(fn = fmin_objective,
                space = space,
                algo = tpe.suggest,
                max_evals = 30,
                trials = trials,
                )
    best_param.append(trials.argmin)
        
    print(f"[Fold {fold + 1}] Training...")
    BF_GBC, pickle_file         = BF_fitting(BF, d_train_list, fold, best_param, pickle_file)

    Prob_matrix                 = BF_predict(BF_GBC, d_val)
    Vote                        = Weighted_Vote(Prob_matrix)
    CV_MCC                      = CV_evaluation(d_val, Vote)  
                  
    print(f"Fold {fold + 1} MCC:\n{CV_MCC}\n")

print("Testing...") 
all_prob_matrix = final_BF_predict(pickle_file, d_test)
Final_vote      = Weighted_Vote(all_prob_matrix)     
Final_MCC       = CV_evaluation(d_test, Final_vote)
     
end = time.time()
print(f"Final evaluation: {Final_MCC}") 


