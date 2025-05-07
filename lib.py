import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import clone
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer

from imblearn.over_sampling import SMOTE

from typing import List

global_random_state = 42

def merge_param_grid(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if type(a) is list and type(b) is list:
        print("Unknown Case")
        quit()
    if type(a) is list:
        merged_param_grid = []
        for grid in a:
            merged = grid.copy()
            merged.update(b)
            merged_param_grid.append(merged)
        return merged_param_grid
    elif type(b) is list:
        merged_param_grid = []
        for grid in b:
            merged = a.copy()
            merged.update(grid)
            merged_param_grid.append(merged)
        return merged_param_grid
    else:   
        return {**a, **b}
        
def print_time_report(grid_search):
    """
    Prints a time report of the training times for each hyperparameter, 
    aggregated across all other hyperparameters. Does not currently work with
    multiple nested lists of hyperparameter grid dictionaries (only one list nesting level)
    """
    results = pd.DataFrame(grid_search.cv_results_)
    #print(grid_search.param_grid)
    if type(grid_search.param_grid) is list:
        for (i, group) in enumerate(grid_search.param_grid):
            print(f"\n====== Training Time by Group {i+1} ======")
            for param in group:
                print(f"\n=== Training Time by '{param}' ===")
                #print(results)
                print(
                    results
                    #Filter by all parameters in the group
                    .loc[results['params'].apply(lambda x: all(k in x for k in group.keys()))]
                    .groupby(f'param_{param}')['mean_fit_time']
                    .mean()
                    .sort_values()
                )
    else:
        for param in grid_search.param_grid:
            print(f"\n=== Training Time by '{param}' ===")
            print(
                results
                .groupby(f'param_{param}')['mean_fit_time']
                .mean()
                .sort_values()
            )