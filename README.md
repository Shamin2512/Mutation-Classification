# SAAPpred with XGB and SKL
## Gradient Boosting and Random Forests for mutation classification
### Files
- HumanVat.txt - Mutation dataset
- pd.csv - Full Pathogenic Deviation data
- snp.csv - Full Single Nucleotide Polymorphism data
- AC_dataset.csv - 3000 protein subsample dataset (PD and SNP combined)
- seed.txt - Seed for experiment reproducibility (changes when running DatasetProcess.py) 
- README.md - readme file

### Directories
- Dev - Folder for scripts/ notebooks used to develop predictor 

### Scripts and notebooks
- DatasetProcess.py - Pre-process dataset script
- DatasetProcess.ipynb - Pre-process dataset notebook

- DatasetScale.py - Scale dataset script
- DatasetScale.ipynb - Scale dataset notebook

- DatascaleSplit.py - Split dataset into training and testing set script
- DatascaleSplit.ipynb - Split dataset into training and testing set notebook

- PDB2AC.py - AC_dataset.csv pre-processing script
- PDB2AC.ipynb - AC_dataset.csv pre-processing notebook 

- XGB_predictor.py - XGB gradient boosting prediction script
- XGB_MutationClassification.ipynb XGB prediction notebook

- SKL_predictor.py - SKL random forests prediction script
- MutationClassification.ipynb SKL prediction notebook

### Running predictor
1) Run "DatasetProcess.py" and enter datasets. Returns "Dataset_NoFeature.csv" and "Dataset_Feature.csv"
2) Run "DatasetScale.py" and enter "Dataset_NoFeature.csv". Returns "SDataset.csv" and "MMDataset.csv"
3) Run "DatasetSplit.py" and enter chosen dataset (SDataset.csv" for standard scaled, "MMDataset.csv" for min-max scaled, or "Dataset_NoFeature.csv" for no scaling). Returns "Training_Set.csv" and "Testing_Set.csv"
4) Run XGB_predictor.py for gradient boosting, or SKL_predictor.py for random forests

