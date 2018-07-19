import sys
sys.path.append('/home/fc443/project/regression_task/')
from exp_uci import split_data, whiten_data
import kernels, time, csv
import numpy as np
import svm

def load_adult():
    dictionaries = [{} for _ in range(15)]
    
    def format_row(row):
        formatted_row = row[:]
        
        for idx, entry in enumerate(row):
            try:
                formatted_row[idx] = float(entry)
            except ValueError:
                # entry is not numeric, insert it in dictionary if necessary and translate it 
                if not entry in dictionaries[idx]:
                    dictionaries[idx][entry] = len(dictionaries[idx])
                formatted_row[idx] = dictionaries[idx][entry]

        X_row = formatted_row[:-1]
        y_row = formatted_row[-1]
        return X_row, y_row

    with open('datasets/adult.data', 'rb') as f:
        X = []
        y = []
        reader = csv.reader(f, delimiter = ',', skipinitialspace=True)
        for row in reader:
            if len(row) > 0:
                X_row, y_row = format_row(row)

                X.append(X_row)
                y.append(y_row)
        
        X = np.matrix(X)
        y = np.array(y)
    print dictionaries, X[0]
    return X, y

"""
def grid_search(X, y, scale):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    grid = np.ones((len(C_range), len(gamma_range)))

    def cv_error(c, g):
        X_train, y_train, X_cv, y_cv = split_data(X, y)
        y_cv_fit = svm.fit_from_kernel_gen(X_train, y_train, X_cv, lambda a, b: kernels.gaussian_kernel(a, b, scale), c, g)
        return np.linalg.norm(y_cv_fit - y_cv) / y_cv.shape[0]

    for i in range(len(C_range)):
        for j in range(len(gamma_range)):
            grid[i, j] = cv_error(C_range[i], gamma_range[j])
        print(grid)

    (best_i, best_j) = np.unravel_index(np.argmax(grid), grid.shape)
    best_C = C_range[best_i]
    best_gamma = gamma_range[best_j]
    
    print('The best parameters for scale %f are (%f, %f) with a score of %f' % (scale, best_C, best_gamma, grid[best_i, best_j]))
"""

def main():
    X, y = load_adult()
    X = whiten_data(X)[0]
    np.random.seed(0)
    X_train, y_train, X_test, y_test = split_data(X, y, 0.9) 
    
    scale = 5.0
    C = 1.0
    # grid_search(X_train, y_train, scale)

    for scale in [2.0, 5.0, 8.0, 11.0, 15.0, 20.0, 25.0, 30.0]:
        y_test_fit = svm.fit_from_feature_gen(X_train, y_train, X_test, lambda a: kernels.iid_gaussian_RFF(a, 256, 0, scale), C)
        print(C, scale, np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0])
    # y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, lambda a, b: kernels.gaussian_kernel(a, b, scale), C)
    # print(C, scale, np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0])

if __name__ == '__main__':
    main()
