from sklearn import svm
import numpy as np

def fit_svm_from_kernel(K, y_train, K_pred, C):
    clf = svm.SVC(kernel = 'precomputed', C = C, verbose = True)
    
    clf.fit(K, y_train)
    return clf.predict(K_pred)

def fit_from_feature_gen(X_train, y_train, X_pred, C, feature_gen):
    PhiX_train = feature_gen(X_train)
    PhiX_pred = feature_gen(X_pred)
    
    clf = svm.LinearSVC(dual = (PhiX_train.shape[0] < PhiX_train.shape[1]), C = C, verbose = False)
    clf.fit(PhiX_train, y_train)

    return clf.predict(PhiX_pred)

def fit_from_kernel_gen(X_train, y_train, X_pred, C, kernel_gen):
    clf = svm.SVC(kernel = kernel_gen, C = C, verbose = True)
    
    clf.fit(X_train, y_train)
    return clf.predict(X_pred)