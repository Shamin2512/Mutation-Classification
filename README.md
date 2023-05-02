# SAAPpred with XGB and SKL
## Gradient Boosting and Random Forests for mutation classification
### Files
- HumanVat.txt - Initial mutation dataset
- pd.csv - Pathogenic Deviation data
- snp.csv - Single Nucleotide Polymorphism data
- seed.txt - Seed for experiment reproducibility (will be changed when running DatasetProcess.py) 
- README.md - readme file

### Directoreis
- Dev - Folder for scripts/ notebooks used to develop predictor

### Scripts and notebooks
- DatasetProcess.py - Pre-process dataset script
- DatasetProcess.ipynb - Pre-process dataset notebook

- DatasetScale.py - Scale dataset script
- DatasetScale.ipynb - Scale dataset notebook

- DatascaleSplit.py - Split dataset into training and testing set script
- DatascaleSplit.ipynb - Split dataset into training and testing set notebook

- XGB_predictor.py - XGB gradient boosting prediction script
- XGB_MutationClassification.ipynb XGB prediction notebook

- SKL_predictor.py - SKL random forests prediction script
- MutationClassification.ipynb SKL prediction notebook

### Running predictor
1) Run "DatasetProcess.py" and enter datasets ("pd.csv" and "snp.csv"). Returns two files: "Dataset_NoFeature.csv" and "Dataset_Feature.csv"
2) Run "DatasetScale.py" and enter "Dataset_NoFeature.csv". Returns "SDataset.csv" and "MMDataset.csv"
3) Run "DatasetSplit.py" and enter chosen dataset (SDataset.csv" for standard scaled, "MMDataset.csv" for min-max scaled, or "Dataset_NoFeature.csv" for no scaling). Returns "Training_Set.csv" and "Testing_Set.csv"
4) Run XGB_predictor.py for gradient boosting, or SKL_predictor.py for random forests

