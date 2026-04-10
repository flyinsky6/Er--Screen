Erα-Screen
This study proposes a complete computational prediction framework for ERα-targeted drug discovery, including ERα bioactivity regression, ADMET classification, and drug-target interaction prediction, using R with caret and rfSA for feature selection and Python for ensemble learning and dual-channel DNN modeling, while applying SMOTE, Gaussian augmentation, and multi-step feature selection to address data imbalance and redundancy, ultimately establishing an integrated model combining XGBoost, LightGBM, and dual-channel DNN for full-flow prediction.
Requirement
R 
caret, corrplot, doParallel
Python
Numerical computing libraries: Numpy 1.20+, pandas 1.3+
Machine learning libraries: scikit-learn 1.0+, xgboost, lightgbm, catboost, imblearn
Deep learning libraries: Tensorflow 2.0+, keras
Bioinformatics libraries: rdkit, biopython
Explainability libraries: shap
Visualization libraries: matplotlib, seaborn
Dataset
This study constructed a DTI dataset containing 1,206 protein-drug pairs. ERα and ADMET models were built based on 1,974 compounds from DrugBank with their SMILES, pIC50 values, and five ADMET descriptors. Massive compound data from KEGG and PubChem were integrated for drug screening.
Features
Small molecules initially included 1,613-dimensional descriptors and 881-dimensional PubChem fingerprints, from which 20 core features were selected via the rfSA algorithm. Drug features were represented by 1,024-dimensional Morgan fingerprints, and 1,244-dimensional multi-type features were extracted from the ERα protein sequence. Feature selection was performed in parallel in R using the caret package with rfSA, and by variance filtering and SelectKBest in Python.
Model 
ERα bioactivity and ADMET prediction: Directly run ER_bioactivity_ADMET.py and load the trained pickle model to achieve batch prediction of ERα bioactivity (pIC50) and ADMET properties (Caco-2, CYP3A4, etc.).
Drug-ERα target interaction prediction: Directly run Drug_target_interaction_model.py and load the dual-channel deep learning model for high-throughput prediction of drug-ERα interactions.
Feature engineering: The R feature engineering script can directly read raw high-dimensional data (e.g., CACO2.csv), perform progressive feature selection, and output a 20-dimensional core feature dataset ready for modeling in Python.
