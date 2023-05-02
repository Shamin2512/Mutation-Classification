# %% [markdown]
# ### Import library

# %% [markdown]
# SAAPpred script that predicts protein pathogenicty from SAAPdap data, using SciKit-Learn random forests. 
# Goal is to predict SNP or PD with MCC > 0.7.
#         
# Best CV models are directly used for final testing

# %%
""" Imports the required libraries and packages """

import pandas as pd                                                              # Data manipulation in dataframes
import numpy as np                                                               # Array manipulation
import pickle
import hyperopt

import random as rd                                                              # Random seed generation
import time                                                                      # Time program run time


import matplotlib.pyplot as plt
from matplotlib.patches import Patch                                             # CV visualise
from matplotlib import colors 

from sklearn.metrics import(
    matthews_corrcoef,                                                           # MCC for evaluation
    confusion_matrix,                                                            # Confusion matrix for classification evalutation
    classification_report                                                        # Return the F1, precision, and recall of a prediction
    )

from sklearn.model_selection import(
    GroupKFold                                                                   # K-fold CV with as groups
        )

from sklearn.ensemble import RandomForestClassifier                              # SK learn API for classificastion random forests

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK                                       # Functions for minimising cost functions
from hyperopt.pyll.base import scope
from functools import partial

np.set_printoptions(precision = 3,threshold=np.inf, suppress=True)               # Full array printing

# %% [markdown]
# ### Split dataset into training and validation sets

# %%
def open_data(file_train, file_test):
    """      
     Returns:    Training_Set     Scaled 80% training set split
                 Testing_Set      Scaled 20% testing set split
            
    Opens the scaled training and testing data
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
# ## Group K-fold CV (outer loop)

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
    groups       = Training_Set['AC Code'].to_list()               #List of proteins for grouping
    
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
# ## Balancing (inner loop)

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

# %% [markdown]
# ### Balance for n folds

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
# ### RFC hyperparameter tuning

# %%
def hyperopt_space():
    """ 
    Returns:   Space         Parameter space for hyperparameter searching

    Define paramater space for Hyperopt to search throug
   """  
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 1250, 2000, 1750)),
        'max_depth': scope.int(hp.quniform('max_depth', 10, 15, 1)),
        'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 6)),
        }
     
    return space


# %%
def objective(params, Input_folds, Output_folds, ValData, Vallabel, seed): 
    """ 
    Input:      params            Search paramaters parsed in fmin function()                     
                Input_folds       List of balanced training feature folds
                Output_folds      List of balanced training label folds
                
    Returns:    loss, status      The MCC from evaluating hyperparameters during search

    Define the model that Hyperopt will optimise hyperparameters for
    """     
    max_depth = params['max_depth']
    min_samples_split = params['min_samples_split']
    n_estimators = params['n_estimators']
    
    RFC = RandomForestClassifier(n_estimators = n_estimators, 
                                 min_samples_split = min_samples_split, 
                                 max_depth = max_depth,
                                 n_jobs = 4,
                                 random_state=seed,
                                 )                
    #Generates and fits a RFC for each training balanced fold
    model = RFC.fit(Input_folds, Output_folds)
    
    pred = model.predict(ValData)
    MCC = matthews_corrcoef(Vallabel, pred.round())
                                                                         
    return {'loss': -MCC, 'status': STATUS_OK}


# %% [markdown]
# ### Train RFC on the trainings folds

# %%
def BF_fitting(BF, Input_folds, Output_folds, fold, final_param, pickle_file, seed): 
    """ 
    Input:      BF                Number of balancing folds                      
                Input_folds       List of balanced training feature folds
                Output_folds      List of balanced training label folds

    Returns:    BF_RFC            List of RFCs trained on each balancing fold

    Create RFC model that returns probability predictions for each fold, using output of Balance_Folds() as training data
    """    
    BF_RFC = []
    
    for i in range(BF):
        
        model = RandomForestClassifier(n_estimators = int(final_param[0]['n_estimators']), 
                                        min_samples_split = int(final_param[0]['min_samples_split']), 
                                        max_depth = int(final_param[0]['max_depth']),
                                        n_jobs = -1,
                                        random_state=seed,
                                        ) 
        model = RandomForestClassifier(random_state=seed) 
        #Generates a RFC for each fold's training data
        
        model.fit(Input_folds[i], Output_folds[i])     #Fits the RFC to each folds' training data
        
        filename = f"RFC_CV_{fold + 1}_model_{i + 1}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
            
        pickle_file.append(filename)
        BF_RFC.append(model)
        
    return BF_RFC, pickle_file


# %% [markdown]
# #### Validate each RFC on validation set, for each fold

# %%
def BF_validate(BF_RFC, ValData):
    """ 
    Input:      BF_RFC          List of RFCs trained on balancing folds
                ValData         Unseen validation features from CV fold
                
    Returns:    Prob_matrix     List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the validation set.
    """
    
    Prob_matrix = []
    
    for i in range(len(BF_RFC)):
        Prob = BF_RFC[i].predict_proba(ValData)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
        Prob_matrix.append(Prob)   
        
    return Prob_matrix

# %% [markdown]
# ### Weighted voting

# %%
def Weighted_Vote(Prob_matrix):
    """ 
    Input:      Prob_matrix     List of arrays. 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability

    Returns:    Final_vote      Weighted vote classification

    Calculate the final predictions with weighted vote using confidence scores. 
    Sc = (S0 -T)/(1-T) if S0> T
    Sc = (T-S0)/T if S0 < T
    """
    Sc_SNP = []
    Sc_PD = []
    
    for i in range(len(Prob_matrix)):
        Sc_SNP.append(Prob_matrix[i][:,0])
        Sc_PD.append(Prob_matrix[i][:,1])

    Sum_SNP   = np.sum(Sc_SNP, axis = 0)     #Sum of all SNP confidence scores. 1D Array
    Sum_PD    = np.sum(Sc_PD, axis = 0)      #Sum of all PD confidence scores. 1D Array
    
    
    Vote_arr  = [] 

    for i in range(len(Sum_PD)):
        if Sum_PD[i] >= Sum_SNP[i]:
            Vote_arr.append([1])                #Append PD classifications to list
        elif Sum_SNP[i] > Sum_PD[i]:
            Vote_arr.append([0])                #Append SNP classifications to list

    Final_vote = np.stack(Vote_arr)             #Converts list of arrays to a 2D array
    Final_vote = Final_vote.ravel()             #Flattens 2D array to 1D array

    return(Final_vote)                          #Returns the CV votes


# %%
def CV_evaluation(Vallabel, Final_vote):
    """ 
    Input:      d_val             Validation data as Dmatrix
                Final_vote        Weighted vote classification
                
    Evaluates a CV fold's trained model with MCC
    """
    Output_pred = Final_vote
    TrueLabel   = Vallabel
        
    CV_MCC = matthews_corrcoef(TrueLabel, Output_pred)

    return CV_MCC

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
    prefix = f"RFC_"
        
    for file in pickle_file:
        if file.startswith(prefix):
            with open(file, "rb") as f:
                    model = pickle.load(f)
                    Prob = model.predict_proba(TestData)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
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
        
    Sc_SNP = []
    Sc_PD = []
    
    for i in range(len(Prob_matrix)):
        Sc_SNP.append(Prob_matrix[i][:,0])
        Sc_PD.append(Prob_matrix[i][:,1])
            
    Sum_SNP = np.sum(Sc_SNP, axis = 0)     #Sum of all SNP confidence scores. 1D Array
    Sum_PD  = np.sum(Sc_PD, axis = 0)      #Sum of all PD confidence scores. 1D Array
                                                    
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


# %%
def final_model_train(BF, Input_folds, Output_folds, final_params, final_files, seed): 
    """ 
    Input:      BF                Number of balancing folds                      
                Input_folds       List of balanced training feature folds
                Output_folds      List of balanced training label folds

    Returns:    BF_RFC            List of RFCs trained on each balancing fold  
    
    Retrains a final model using the best params from the best performance CV on the entire training data
    """
    final_BF_RFC = []
    for i in range(BF):
        
        model = RandomForestClassifier(n_estimators = int(final_params['n_estimators']), 
                                        min_samples_split = int(final_params['min_samples_split']), 
                                        max_depth = int(final_params['max_depth']),
                                        n_jobs = -1,
                                        random_state=seed,
                                        ) 
        
        model.fit(Input_folds[i], Output_folds[i])    
            
        final_BF_RFC.append(model)
        #Generates a RFC for each fold's training data
    
        
        filename = f"final_model_{i + 1}.pkll"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
            
        final_files.append(filename)
        
    return final_BF_RFC


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
final_files = []
CV_score = []
final_param = []

print("Opening dataset...")
TrainData, TrainLabels, TestData, TestLabels = learning_data(Training_Set, Testing_Set)   

print("Performing Group fold cross validation...")           
IT_list, LT_list, IV_list, LV_list = CV(Training_Set)     

for fold in range(len(IT_list)):       

    inData                      = IT_list[fold]
    classData                   = LT_list[fold]
    ValData                     = IV_list[fold]
    Vallabel                    = LV_list[fold]

    print(f"[Fold {fold + 1}] Balancing...")
    minClass, minSize, maxSize  = find_minority_class(classData)   
    BF                          = Balance_ratio(maxSize, minSize)                        
    Input_folds, Output_folds   = Balance_Folds(BF, inData, classData, minClass, minSize)

    print(f"[Fold {fold + 1}] CV Training...")

    space                   = hyperopt_space()
    trials                  = Trials()

    rd_fold                 = rd.randrange(BF)
    fmin_objective          = partial(objective,
                                        Input_folds    = Input_folds[rd_fold],
                                        Output_folds   = Output_folds[rd_fold],
                                        ValData        = ValData,
                                        Vallabel       = Vallabel,
                                        seed           = seed,
                                    )     
                    
    best                    = fmin(fn = fmin_objective,
                                    space             = space,
                                    algo              = tpe.suggest,
                                    max_evals         = 30,
                                    trials            = trials,
                                    )

    final_param.append(trials.argmin)

    BF_RFC, pickle_file                      = BF_fitting(BF, Input_folds, Output_folds, fold, final_param, pickle_file, seed)
    Prob_matrix                              = BF_validate(BF_RFC, ValData)

    Final_vote                               = Weighted_Vote(Prob_matrix)
    CV_MCC                                   = CV_evaluation(Vallabel, Final_vote)  
    CV_score.append(CV_MCC)
                
    print(f"Fold {fold + 1} MCC:\n{CV_MCC}\n")

best_fold = (CV_score.index(max(CV_score)) + 1)
final_params = final_param[(best_fold - 1)]

print("Testing...") 
Prob_matrix = final_BF_predict(pickle_file, TestData)
MCC_final = final_evaluation(Prob_matrix, TestLabels) 
print(MCC_final) 

print(f"Re-training final model...\n")
inData = TrainData
classData = TrainLabels
ValData = TestData
Vallabel = TestLabels
minClass, minSize, maxSize  = find_minority_class(classData)   
BF                          = Balance_ratio(maxSize, minSize)                        
Input_folds, Output_folds   = Balance_Folds(BF, inData, classData, minClass, minSize)

final_BF_RFC                = final_model_train(BF, Input_folds, Output_folds, final_params, final_files, seed)

print("Testing...") 
final_prob_matrix                 = BF_validate(final_BF_RFC, TestData)
final_final_vote                  = Weighted_Vote(final_prob_matrix)
final_MCC                         = CV_evaluation(TestLabels, final_final_vote)

Score_list.append(final_MCC)
#loop     
end = time.time()
print(f"Final evaluation:{np.mean(Score_list)} \u00B1 {np.std(Score_list)}\n\nLowest score:{min(Score_list)}\nHighest score:{max(Score_list)}\n\nRun time: {end-start}")


