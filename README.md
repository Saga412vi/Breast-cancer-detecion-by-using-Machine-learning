# Breast-cancer-detecion-by-using-Machine-learning
ML Project
# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
type(cancer_dataset)
# keys in dataset
cancer_dataset.keys()
# featurs of each cells in numeric format
cancer_dataset['data']
# malignant or benign value
cancer_dataset['target']
# target value name malignant or benign tumor
cancer_dataset['target_names']
# description of data
print(cancer_dataset['DESCR'])
# name of features
print(cancer_dataset['feature_names'])
