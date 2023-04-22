# %% [markdown]
# SAAPpred script that predicts protein pathogenicty from SAAPdap data, using XG Boost. Used on on "AC_dataset" (imbalance 2:1: ~2200 PD and ~1100 SNP) and "Dataset_NoFeature" (imbalance 18:1: ~460000 PD and ~25000 SNP)
# Goal is to predict SNP or PD with MCC > 0.7.
#         
# Scale branch

# %% [markdown]
# ### Import library

# %%
""" Imports the required libraries and packages """

import pandas as pd                                                              # Data manipulation in dataframes
import numpy as np                                                               # Array manipulation
import xgboost as xgb                                                            # Gradient boosting package
import pickle                                                                    # Saving/loading GBM files
# import hyperopt

import random as rd                                                              # Random seed generation
import time                                                                      # Time program run time
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from xgboost import plot_importance

from sklearn.metrics import(
    matthews_corrcoef,                                                           # MCC for evaluation
    # balanced_accuracy_score, #hyperparameter evaluation
    # f1_score,  #hyperparameter evaluation
    accuracy_score,
    confusion_matrix,                                                            # Confusion matrix for classification evalutation
    classification_report                                                        # Return the F1, precision, and recall of a prediction
    )

from sklearn.model_selection import(
    train_test_split,                                                            # Splits data frame into the training set and testing set
    # GridSearchCV,  # Searches all hyperparameters
    # RandomizedSearchCV, # Searches random range of hyperparameters
    GroupKFold                                                                   # K-fold CV with as groups
        )

from sklearn.utils import shuffle
# from sklearn.ensemble import RandomForestClassifier                            # SK learn API for classificastion random forests

# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK                                       # Functions for minimising cost functions
# from hyperopt.pyll.base import scope

np.set_printoptions(precision = 3,threshold=np.inf, suppress=True)               # Full array printing

# %% [markdown]
# ### Split dataset into training and validation sets

# %%
# def Train_Test_Split(file):
#     """      
#     Input:      file             Pre-processed dataset done by PDB2AC script

#     Returns:    Training_Set     80% training set split
#                 Testing_Set      20% testing set split
                
#     80% training and 20% testing split. Splits are shuffled randomly and index reset
#     """
#     AC_dataset                  = pd.read_csv(file, index_col=0)  
#     Training_Set                = AC_dataset
        
#     Training_Set, Testing_Set   = train_test_split(AC_dataset,train_size = 0.8)
        
#     Training_Set.reset_index(drop=True, inplace = True)         #Drop index to avoid training on index values
#     Testing_Set.reset_index(drop=True, inplace = True)          #Reset index after splitting for compatability with group fold CV
    
#     Training_Set                = Training_Set.sample(frac = 1) #Shuffle data after splitting
#     Testing_Set                 = Testing_Set.sample(frac = 1)
    
#     # Training_Set.to_csv("Training_Set.csv", index=False)
#     # Testing_Set.to_csv("Testing_Set.csv", index=False)

#     Training_Set = pd.read_csv('Scaled_Training_Set.csv')
#     Testing_Set = pd.read_csv('Scaled_Testing_Set.csv')

    
#     return Training_Set, Testing_Set

# %%
def open_data():
    """      
     Returns:    Training_Set     Scaled 80% training set split
                 Testing_Set      Scaled 20% testing set split
            
    Opens the scaled training and testing data
    """
    Training_Set = pd.read_csv("Training_Set.csv", index_col = 0)
    Testing_Set = pd.read_csv("Testing_Set.csv",index_col = 0)
    
    return Training_Set, Testing_Set

# %%
def test_dmatrix(Testing_Set):
    """      
    Input:      Testing_Set      20% testing set split

    Returns:    d_test           Testing data in dmatrix 
                TestData         Testing features 
                TestLabels       Testing labels
            
    Testing set as dmatrix for XGBoost API compatabillity
    """
    TestData     = Testing_Set.drop(['AC Code','dataset'], axis =1)  
    TestLabels   = Testing_Set['dataset']                                
    
    d_test = xgb.DMatrix(TestData, TestLabels)
    
    return (d_test, TestData, TestLabels)

# %% [markdown]
# ### Initial evaluation

# %%
def test(Training_Set, d_test):
    """ 
    Input:      Training_Set     80% training set split
                d_test           Testing data in dmatrix

    Evaluate training data before CV and balancing. Gradient boosting for prediction on the test data. 
    True values are testing data class labels
    """    
    TrainData     = Training_Set.drop(['AC Code','dataset'], axis =1)  
    TrainLabels   = Training_Set['dataset']
    d_train = xgb.DMatrix(TrainData, TrainLabels)

    params = {
    'booster': 'gbtree',
    'objective': 'binary:hinge', 
    }
    XGB_initial = xgb.train(params, d_train)
    
    Output_pred = XGB_initial.predict(d_test)
    print(f"              **Initial Evaluation**")
    print(f"Confusion Matrix:\n {confusion_matrix(d_test.get_label(), Output_pred)}")
    print(f"MCC              {matthews_corrcoef(d_test.get_label(), Output_pred)}\n")


# %% [markdown]
# # Outer Loop: Group Fold CV

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
# # Inner Loop:

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
    BF = (2 * round(Divide)) + 1    #Double ratio to nearest integer
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

    Converts the balanced training data and validation data into Dmatrix for model training and evaluation
    """

    d_train_list =[]
    
    for i in range(BF):
        d_train = xgb.DMatrix(Input_folds[i], Output_folds[i])      #Create DMatrix for each training balanced fold
        d_train_list.append(d_train)
    d_val = xgb.DMatrix(ValData, Vallabel)

    return (d_train_list, d_val)


# %%
def MCC_eval_metric(pred, d_val):
    """ 
    Input:      pred              Prediction from a boosted tree during training
                d_val             Validation data as Dmatrix
    
    Returns:    mcc               The MCC from a boosted tree round

    MCC as a custom evaluation metric to evaluate model training during cross validation. This is different from the final weighted vote evaluation.
    """
    true_label = d_val.get_label()   
    pred_label = np.round(pred) 
    
    return 'mcc', matthews_corrcoef(pred_label, true_label)

# %%
# def hyperopt_space():
#     """ 
#     Returns:    Space         Parameter space for hyperopt searching

#     Define paramater psace for hyperopt tuning
#    """  
# #   params = {
# #     'booster': 'gbtree',
# #     'objective': 'binary:logistic', 
# #     # 'learning_rate': 0.3,
# #     # 'max_depth': 5,
# #     }
# #   for i in range(BF):        
# #     BF_GBC_HP = xgb.cv(
# #         params,
# #         d_train_list[i],
# #         nfold = 5,
# #         num_boost_round= 500,
# #         early_stopping_rounds= 20,
# #         custom_metric = CM, 
# #         as_pandas=True,
# #     )
  
# #   return(BF_GBC_HP) 

#     space = {
#         'max_depth': hp.quniform('max_depth', 3, 18),
#         # 'eta': hp.loguniform('eta', -7, 0),
#         # 'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
#         # 'gamma': hp.loguniform ('gamma', -10, 10),
#         # 'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
#         # 'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
#         }     
    
#     return space


# %%
# def BF_optimise(BF, d_train_list, d_val, space): 
#     """ 
#     Input:      BF                Number of balancing folds                      
#                 d_train_list      List of balanced training feature folds in DMatrix
#                 d_val             Validation data as Dmatrix
                
#     Returns:    BF_GBC            List of gradient boosted models trained on each balancing fold

#     Create GBC model that returns probability predictions for each fold, using output of Balance_Folds() as training data (as Dmatrix)
#     """     
    
#     params = {
#         'booster': 'gbtree',
#         'objective': 'binary:logistic',
#         'eval_metric': 'error',
#         'disable_default_eval_metric': 0,
#         'verbosity': 1,
#         'max_depth': space['max_depth'],
#               } 
#         # 'gamma': space['gamma'],
#         # 'alpha': space['alpha'],
#         # 'lambda': space['lambda'],
#         # 'colsample_bytree': space['colsample_bytree'],
#         # 'min_child_weight': space['min_child_weight'],
    
#     BF_models = []
#     # for fold_i in range(BF):
#     d_train = d_train_list[0]                              #Dmatrix for each balanced fold
#     BF_models = xgb.train(params,
#                           d_train,
#                           num_boost_round = 1000,
#                           evals  = [(d_val,'Model')],
#                           verbose_eval = False,
#                           early_stopping_rounds = 10,
#                           )
#                                                                  #Generates and fits a GBC for each training balanced fold
#     # pred_list = []
#     # for i in len(range(BF_models)):
#     pred = BF_models.predict(d_val)
#     accuracy_score = (d_val.get_label(), pred > 0.5)
                                                                      
#     return accuracy_score

# %%
# def BF_fitting(BF, d_train_list, d_val, MCC_eval_metric, fold): 
def BF_fitting(BF, d_train_list, d_val, MCC_eval_metric, fold, pickle_file): 

    """ 
    Input:      BF                Number of balancing folds                      
                d_train_list      List of balanced training feature folds in DMatrix
                d_val             Validation data as Dmatrix
                
    Returns:    BF_GBC            List of gradient boosted models trained on each balancing fold

    Create GBC model that returns probability predictions for each fold, using output of Balance_Folds() as training data (as Dmatrix)
    """     
        
    params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'binary:logistic', 
    'eta': 0.8,
    'max_depth': 9,
    # 'reg_alpha': 0.01,
    # 'gamma': 0.01,
    'disable_default_eval_metric': 1,
    'verbosity': 0,
    'nthreads': -1,
    # 'eval_metric':['error'],
    } 
    
    
    BF_GBC = []
    
    for fold_i in range(BF):
        d_train = d_train_list[fold_i]                              #Dmatrix for each balanced fold
        model = xgb.train(params,                                   #Generates and fits a GBC for each training balanced fold
                          d_train, 
                          num_boost_round = 1500,
                          evals  = [(d_val,'Model')],
                          verbose_eval = False,               #Print evaluation metrics every 50 trees
                          early_stopping_rounds = 100,
                          custom_metric = MCC_eval_metric,
                          )
        
        filename = f"CV_{fold + 1}_model_{fold_i + 1}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
            
        BF_GBC.append(model)
        pickle_file.append(filename)
        
    return BF_GBC, pickle_file
    # return BF_GBC

# %% [markdown]
# ### Validation

# %%
def BF_predict(BF_GBC, d_val):
    """ 
    Input:      BF_RFC            List of RFCs trained on balancing folds
                d_val             Validation data as Dmatrix

                
    Returns:    Prob_matrix     List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the validation set.
    """
    
    Prob_matrix = []
    for i in range(len(BF_GBC)):
        Prob = BF_GBC[i].predict(d_val)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
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
                fold              CV fold number, defined in main program

    Evaluates a CV fold's trained model with a classificaiton report, including the MCC
    """
    Output_pred = Final_vote
    TrueLabel   = d_val.get_label()
        
    # print(f"-----------------------------------------------------\n              ***CV Fold Evaluation***\n")
    # print(f"Confusion Matrix:\n {confusion_matrix(TrueLabel, Output_pred)}")
    # print(f"{classification_report(TrueLabel, Output_pred)}\nMCC                  
    CV_MCC = print(f"CM:\n{confusion_matrix(TrueLabel, Output_pred)}\nMCC: {matthews_corrcoef(TrueLabel, Output_pred)}\n")
    
    return CV_MCC

# %%
# def fold_models(BF_GBC, all_model_list):
#     """ 
#     Input:      BF_RFC            List of 5 GBMs trained on balanced folds
#                 all_model_list    List that will contain all fold GBM models
                
#     Returns:    all_model_list    List of all GBM models from all fold. 25 total
    
#     Adds all GBM models for a given fold to a list for final testing.
#     """
#     all_model_list.append(BF_GBC)
    
#     return all_model_list

# %% [markdown]
# # Outer Loop: Final evaluation 

# %%
def fold_predict(d_test, pickle_file):
# def fold_predict(all_model_list, d_test):

    """ 
    Input:      all_model_list    List of all GBM models from all fold. 25 total
                d_test            Unseen testing data as Dmatrix

                
    Returns:    fold_prob_matrix  List of arrays. Each item is 2D matrix where the 1st dimension is each subset in balancing fold, 
                                  2nd dimension is predicted probability
    
    Predicts the probabilty for every datapoint in the testing set.
    """
    
    all_prob_matrix = []
    all_models = []
    
    for file in pickle_file:
        with open(file, "rb") as f:
            model = pickle.load(f)
            all_models.append(model)
            
    for i in range(len(all_models)):
        Prob = all_models[i].predict(d_test)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
        all_prob_matrix.append(Prob)   

    # for i in range(len(all_model_list)):
    #     for j in range(len(all_model_list[i])):
    #         Prob = all_model_list[i][j].predict(d_test)     #Predicts the probability of an instance belonging to the major/ positive class (PD/ 1). Output has shape (n_predictions,)
    #         all_prob_matrix.append(Prob)   
        
    return all_prob_matrix

# %%
def final_evaluation(all_prob_matrix, TestLabels):
    """ 
    Input:      all_prob_matrix    List of all predicted probabilites from all optimised models
                TestLabels         True labels from unseen 20% testing data

    Returns:    MCC_final          Final MCC evaluation

    Calculate the final weighted vote using confidence scores (Sc) from all_prob_matrix. Then evaluates votes agains true labels to give the final MCC
    """
    
    # flat_list = [matrix for proba in all_prob_matrix for matrix in proba]
    
    PD_prob_matrix = all_prob_matrix

    SNP_prob_matrix = []
    for i in range(len(all_prob_matrix)):                 #SNP probabilites are 1 - (PD probabilites)
        sub = 1 - all_prob_matrix[i]
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


# %%
def plot(Score_list):
     """ 
     Input:      Score_list        List of MCC scores

     Plots the MCCs of 15 runs, and the average
     """
     fig, ax = plt.subplots(figsize=(16,10), dpi= 65)
     x_axis = range(len(Score_list))
     y_axis = Score_list

     plt.scatter(x_axis, y_axis, color = 'teal')
     plt.axhline(y=np.nanmean(Score_list), color = 'red', linestyle = 'dotted', linewidth = '1', label ='Avg')
     plt.xlabel('Run Number')
     plt.ylabel('MCC')
     plt.legend()
     plt.show()

# %% [markdown]
# ### Main Program

# %%
start = time.time()

Score_list = []
# all_model_list = []
pickle_file = []

# for i in range(0,15):
print("Opening dataset...")
Training_Set, Testing_Set          = open_data()
d_test, TestData, TestLabels = test_dmatrix(Testing_Set)   

# test(Training_Set, d_test)
print("Performing Group fold validation...")
            
IT_list, LT_list, IV_list, LV_list = CV(Training_Set)     

for fold in range(len(IT_list)):          
    inData = IT_list[fold]
    classData = LT_list[fold]
    ValData = IV_list[fold]
    Vallabel = LV_list[fold]

    #Validation
    print(f"[Fold {fold}] Balancing...")
    minClass, minSize, maxSize  = find_minority_class(classData)   
    BF                          = Balance_ratio(maxSize, minSize)                        
    Input_folds, Output_folds   = Balance_Folds(BF, inData, classData, minClass, minSize)
    d_train_list, d_val         = GBM_dmatrix(BF, Input_folds, Output_folds, ValData, Vallabel)
    
    # space = hyperopt_space()
    # accuracy_score = BF_optimise(BF, d_train_list, d_val, space)
    # trials = Trials()
    # best = fmin(fn = accuracy_score,
    #             space = space,
    #             algo = tpe.suggest,
    #             max_evals = 140,
    #             trials = trials,
    #             )
    print(f"[Fold {fold}] Training...")
    BF_GBC, pickle_file         = BF_fitting(BF, d_train_list, d_val, MCC_eval_metric, fold, pickle_file)

    Prob_matrix                 = BF_predict(BF_GBC, d_val)
    Final_vote, Sum_PD, Sum_SNP = Weighted_Vote(Prob_matrix)
    # CV_MCC = CV_evaluation(d_val, Final_vote)                        #prints classification report for all 5 folds
    # CV_MCC
    # all_model_list = fold_models(BF_GBC, all_model_list)
    
#Testing
print("Testing...") 
all_prob_matrix = fold_predict(d_test, pickle_file) 

MCC_final = final_evaluation(all_prob_matrix, TestLabels) 
print(MCC_final)   
Score_list.append(MCC_final) 
     
end = time.time()
# plot(Score_list)
print(f"Final evaluation:{np.mean(Score_list)} \u00B1 {np.std(Score_list)}\n\nLowest score:{min(Score_list)}\nHighest score:{max(Score_list)}\n\nRun time: {end-start}")


