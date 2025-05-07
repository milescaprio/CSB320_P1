# 1: Create Model and Pipeline Presets

from lib import *

def get_preprocessing_pipe(scaling = True, preprocessing = "PCA", resampling = True):
    pipe = []
    if resampling:
        pipe.append( ('SMOTE', SMOTE(random_state = global_random_state)) )
    if scaling:
        pipe.append( ('Scaler', StandardScaler()) )
    if preprocessing == "PCA":
        pipe.append( ('PCA', PCA(n_components=2)) )
    if preprocessing == "ICA":
        pipe.append( ('ICA', FastICA(n_components=2)) )
    return pipe

def get_knn_pipe(base_pipe : List = None):
    """Returns a pipeline for KNN classifier with optional scaling and preprocessing."""
    pipe = [
        ('KNN', KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=-1))
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)
    
def get_naive_bayes_pipe(base_pipe : List = None):
    """Returns a pipeline for Naive Bayes classifier with optional scaling and preprocessing."""
    pipe = [
        ('clf', GaussianNB())
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)

def get_log_reg_pipe(base_pipe = None):
    """Returns a pipeline for Logistic Regression classifier with optional scaling and preprocessing."""
    pipe = [
        ('LogisticRegression', LogisticRegression(max_iter=1000))
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)

def get_decision_tree_pipe(base_pipe = None):
    """Returns a pipeline for Decision Tree classifier with optional scaling and preprocessing."""
    pipe = [
        ('DecisionTree', DecisionTreeClassifier())
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)

def get_lin_svm_pipe(base_pipe = None):
    """Returns a pipeline for SVM classifier with optional scaling and preprocessing."""
    pipe = [
        ('SVM', LinearSVC(max_iter=10000, dual=False, random_state = global_random_state))
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)

def get_kernel_svm_pipe(base_pipe = None):
    """Returns a pipeline for SVM classifier with optional scaling and preprocessing."""
    pipe = [
        ('SVM', SVC(kernel='linear', probability=False, random_state = global_random_state))
    ]
    if base_pipe:
        pipe = base_pipe + pipe
    return Pipeline(pipe)