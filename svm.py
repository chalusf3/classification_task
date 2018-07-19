from sklearn import svm
import numpy as np

def fit_svm_from_kernel(K, y_train, K_pred):
    clf = svm.SVC(kernel = 'precomputed')
    
    clf.fit(K, y_train)
    return clf.predict(K_pred)

def fit_from_feature_gen(X_train, y_train, X_pred, C, feature_gen):
    PhiX_train = feature_gen(X_train)
    PhiX_pred = feature_gen(X_pred)
    
    clf = svm.LinearSVC(dual = (PhiX_train.shape[0] < PhiX_train.shape[1]), verbose = False)
    clf.fit(PhiX_train, y_train)

    return clf.predict(PhiX_pred)

def fit_from_kernel_gen(X_train, y_train, X_pred, C, kernel_gen):
    K = kernel_gen(X_train, X_train)
    K_pred = kernel_gen(X_pred, X_train)
    
    clf = svm.SVC(kernel = kernel_gen, verbose = False)
    clf.fit(K, y_train)
    return clf.predict(K_pred)
