# Titanic Dataset - Data Cleaning and Data Preprocessing

This repository contains:
- `preprocessing.py` : Python script to clean and preprocess the Titanic dataset
- `titanic.csv` : Titanic dataset file

## What the script does
1. Handles missing values (`Age`, `Fare`, `Embarked`)
2. Encodes categorical features (`Sex`, `Embarked`, `Pclass`)
3. Removes outliers using the IQR method
4. Scales numerical features (`Age`, `Fare`) with StandardScaler
5. Displays boxplots to visualize distributions

## How to run
Make sure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
