# 2: Creating Grid Search Presets

from lib import *


def get_preprocessing_grid_search(pipe):
    """
    Returns a GridSearchCV *argument* for the preprocessing pipeline. 
    Should be combined with another pipeline component.
    """
    # Define the parameter grid for PCA/ICA and scaling
    param_grid = {
        'Scaler': [StandardScaler(), None],
    }
    if 'PCA' in pipe.named_steps:
        param_grid['PCA__n_components'] = [2, 3, 4, 5]
        param_grid['PCA__random_state'] = [42]
    if 'ICA' in pipe.named_steps:
        param_grid['ICA__n_components'] = [2, 3, 4, 5]
        param_grid['ICA__random_state'] = [42]

    return param_grid


def get_knn_grid_search(pipe, base_grid=None, scoring='accuracy',
                        cv=StratifiedKFold(5)):
    """Returns a GridSearchCV object for KNN classifier with a parameter grid."""
    # Define the parameter grid for KNN
    param_grid = {
        'KNN__n_neighbors': [3, 5, 7, 9],
        'KNN__weights': ['uniform', 'distance'],
        'KNN__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    knn_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   scoring=scoring, cv=cv, n_jobs=-1,
                                   return_train_score=True)

    return knn_grid_search


def get_naive_bayes_grid_search(pipe, base_grid=None, scoring='accuracy', 
                                cv=StratifiedKFold(5)):
    """
    Returns a GridSearchCV object for Naive Bayes classifiers with a 
    parameter grid.
    """
    param_grid = [{
        'clf': [GaussianNB()],
        'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
    },
        # {
        #     'clf': [MultinomialNB()],
        #     'clf__alpha': [0.5, 1.0, 1.5]
        # },
        # Not using because it requires non-negative features
        {
        'clf': [BernoulliNB()],
        'clf__alpha': [0.5, 1.0],
        'clf__binarize': [0.0, 0.5]
    }]

    param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    nb_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                  scoring=scoring, cv=cv, n_jobs=-1,
                                  return_train_score=True)

    return nb_grid_search


def get_log_reg_grid_search(pipe, base_grid=None, scoring='accuracy', cv=StratifiedKFold(5)):
    """Returns a GridSearchCV object for Logistic Regression with a 
    parameter grid."""
    # Define the parameter grid for Logistic Regression
    param_grid = {
        'LogisticRegression__C': [0.01, 0.1, 1, 10, 100],
        'LogisticRegression__penalty': ['l2', None],
        'LogisticRegression__solver': ['lbfgs', 'liblinear']
    }

    param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    log_reg_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                       scoring=scoring, cv=cv, n_jobs=1,
                                       return_train_score=True)

    return log_reg_grid_search


def get_decision_tree_grid_search(pipe, base_grid=None, scoring='accuracy', cv=StratifiedKFold(5)):
    """Returns a GridSearchCV object for Decision Tree classifier with a
    parameter grid."""
    # Define the parameter grid for Decision Tree
    param_grid = {
        'DecisionTree__criterion': ['gini', 'entropy'],
        'DecisionTree__max_depth': [None, 5, 10, 15],
        'DecisionTree__min_samples_split': [2, 5, 10],
        'DecisionTree__min_samples_leaf': [1, 2, 4]
    }

    if base_grid:
        param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    dt_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                  scoring=scoring, cv=cv, n_jobs=-1)

    return dt_grid_search


def get_lin_svm_grid_search(pipe, base_grid=None, scoring='accuracy',
                            cv=StratifiedKFold(5)):
    """Returns a GridSearchCV object for LinearSVC classifier with a 
    parameter grid."""
    # Define the parameter grid for SVM
    param_grid = {
        'SVM__C': [0.001, 0.01, 0.1, 1, 10, 50, 100],
        'SVM__penalty': ['l2', 'l1'],
    }

    param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    svm_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   scoring=scoring, cv=cv, n_jobs=-1,
                                   return_train_score=True)

    return svm_grid_search


def get_kernel_svm_grid_search(pipe, base_grid=None, scoring='accuracy',
                               cv=StratifiedKFold(5)):
    """Returns a GridSearchCV object for SVM classifier with a parameter 
    grid."""
    # Define the parameter grid for SVM
    param_grid = [
    {
        'SVM__C': [0.01, 0.1, 1],
        'SVM__kernel': ['linear'],
        'SVM__probability': [False],
        'SVM__gamma': ['scale']
    },
    {
        'SVM__C': [0.01, 0.05, 0.1, 1, 5, 10, 50, 100],
        'SVM__kernel': ['rbf', 'sigmoid'],
        'SVM__probability': [False],
        'SVM__gamma': ['scale']
    },
    {
       'SVM__C': [0.01, 0.1, 1],
       'SVM__kernel': ['poly'],
       'SVM__degree': [2, 3, 4, 5],
       'SVM__probability': [False],
       'SVM__gamma': [1e-9]
    }]

    param_grid = merge_param_grid(base_grid, param_grid)

    # Create a GridSearchCV object
    svm_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   scoring=scoring, cv=cv, n_jobs=-1,
                                   return_train_score=True)

    return svm_grid_search
