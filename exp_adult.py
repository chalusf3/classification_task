import sys, time, csv, pickle, numbers
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from tensorflow.examples.tutorials.mnist import input_data

import svm
sys.path.append('/home/fc443/project/regression_task/')
import kernels
from exp_uci import split_data, whiten_data, algos_generator, plot_runtimes

def load_adult():
    dictionaries = [{} for _ in range(15)]
    
    def format_row(row):
        formatted_row = row[:]
        
        for idx, entry in enumerate(row):
            try:
                formatted_row[idx] = float(entry)
            except ValueError:
                # entry is not numeric, insert it in dictionary if necessary and translate it 
                entry = entry.replace('.', '')
                if not entry in dictionaries[idx]:
                    dictionaries[idx][entry] = len(dictionaries[idx])
                formatted_row[idx] = dictionaries[idx][entry]

        X_row = formatted_row[:-1]
        y_row = formatted_row[-1]
        return X_row, y_row
    with open('datasets/adult.data', 'rb') as f:
        X_train = []
        y_train = []
        reader = csv.reader(f, delimiter = ',', skipinitialspace=True)
        for row in reader:
            if len(row) > 0:
                X_row, y_row = format_row(row)

                X_train.append(X_row)
                y_train.append(y_row)
        
        X_train = np.matrix(X_train)
        y_train = np.array(y_train)
    count = sum([len(dic) for dic in dictionaries])
    with open('datasets/adult.test', 'rb') as f:
        next(f)
        X_test = []
        y_test = []
        reader = csv.reader(f, delimiter = ',', skipinitialspace=True)
        for row in reader:
            if len(row) > 0:
                X_row, y_row = format_row(row)

                X_test.append(X_row)
                y_test.append(y_row)
        
        X_test = np.matrix(X_test)
        y_test = np.array(y_test)
    if count != sum([len(dic) for dic in dictionaries]):
        raise('Error in load_adult, test set contained never before seen attributes.')
        
    X = np.concatenate([X_train, X_test], axis = 0)
    y = np.concatenate([y_train, y_test])
    
    return X, y

def load_covtype():
    with open('datasets/covtype.data', 'rb') as f:
        X = []
        y = []
        reader = csv.reader(f, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1])

    X = np.matrix(X)
    y = np.array(y)

    return X, y

def load_bank():
    # row = age;job;married/single/edu/
    dictionaries = [{} for _ in range(21)]
    dictionaries[3] = {'illiterate'         :0,
                       'basic.4y'           :1,
                       'basic.6y'           :2,
                       'basic.9y'           :3,
                       'unknown'            :4,
                       'high.school'        :5,
                       'professional.course':6, 
                       'university.degree'  :7}
    # dictionaries[9] = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    # dictionaries[10] = {'mon':1, 'tue':2, 'wed':3, 'thu': 4, 'fri': 5}

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
    
    with open('datasets/bank-additional-full.csv', 'rb') as f:
        next(f)
        X = []
        y = []
        reader = csv.reader(f, delimiter = ';', skipinitialspace=True)
        for row in reader:
            if len(row) > 0:
                X_row, y_row = format_row(row)
                X.append(X_row)
                y.append(y_row)
        
        X = np.matrix(X)
        y = np.array(y)
    return X, y

def load_MNIST():
    mnist = input_data.read_data_sets("datasets/MNIST_data/")

    print mnist.train.images.shape, mnist.train.labels.shape
    print mnist.test.images.shape, mnist.test.labels.shape
    # X = mnist.train.images
    # y = mnist.train.labels
    X = np.concatenate([mnist.train.images, mnist.test.images], axis = 0)
    y = np.concatenate([mnist.train.labels, mnist.test.labels], axis = 0)

    return X, y

def classification_error_n_rff(data_name, algos, X_train, y_train, X_test, y_test, C):
    timing = False
    if timing:
        n_seeds = 100
    else:
        n_seeds = 5
    
    if data_name == 'adult':
        n_rffs = [4, 8, 12, 16, 20] + range(24, 264 + 1, 24)
        n_rffs = [216]
    elif data_name == 'bank':
        n_rffs = [4, 8, 12, 16, 20] + range(24, 264+1, 24)
    elif data_name == 'MNIST':
        n_rffs = range(56, 1+784, 112) + range(896, 2017, 224)

    for algo_name, feature_gen_handle in algos.items():
        print 'Starting %s for data %s' % (algo_name, data_name)
        errors = defaultdict(list)
        errors['runtimes'] = defaultdict(list)
        for n_rff in n_rffs:
            errors[n_rff] = np.zeros(n_seeds)
            start_time = time.clock()
            for seed in range(n_seeds):
                y_test_fit = svm.fit_from_feature_gen(X_train, y_train, X_test, C, lambda raw_feature: feature_gen_handle(raw_feature, n_rff, seed))
                errors[n_rff][seed] = np.mean(np.abs(y_test_fit != y_test))
            errors['runtimes'][n_rff] = (time.clock() - start_time) / n_seeds
            print('{} {} \t{} \t{:.4}sec'.format(algo_name, n_rff, np.mean(errors[n_rff]), errors['runtimes'][n_rff]))
    
        if timing:
            filename = 'output/timing/%s_%s_svm.pk' % (data_name, algo_name)
        else:
            filename = 'output/%s_%s_svm.pk' % (data_name, algo_name)
        
        try:
            with open(filename, 'rb') as f:
                old_errors = pickle.load(f)
                if 'runtimes' in old_errors.keys() and not timing:
                    errors['runtimes'] = old_errors['runtimes']
        except IOError:
            print '%s file did not previously exist' % filename
        except EOFError:
            print '%s file was not a pickle file' % filename
        
        if len(n_rffs) > 1:        
            print 'saving in %s' % filename
            with open(filename, 'wb') as f:
                pickle.dump(errors, f)

def classification_error_kernel(data_name, X_train, y_train, X_test, y_test, C, scale):
    if data_name == 'adult':
        n_rffs = [4, 264]
    elif data_name == 'bank':
        n_rffs = [4, 264]

    errors = {}
    errors['runtimes'] = {}
    n_trials = 1
    start_time = time.clock()
    for trial_count in range(n_trials):
        print 'kernel: %d/%d' % (trial_count, n_trials)
        y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, C, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[0]] = np.mean(np.abs(y_test_fit != y_test))
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print 'SE kernel \t{} \t{:.4}sec'.format(errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    if n_trials > 1:
        filename = 'output/timing/%s_exact_gauss_svm.pk' % data_name
    else:
        filename = 'output/%s_exact_gauss_svm.pk' % data_name

    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)

    """
    errors = {}
    errors['runtimes'] = {}
    errors[n_rffs[0]] = 0
    n_trials = 30
    start_time = time.clock()
    for trial_count in range(n_trials):
        y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, C, lambda a, b: np.dot(kernels.iid_gaussian_RFF(a, 264, trial_count, scale), kernels.iid_gaussian_RFF(b, 264, trial_count, scale).T) / 264)
        errors[n_rffs[0]] += np.mean(np.abs(y_test_fit != y_test)) / n_trials
        print 'SE kernel (LibLinear) \t{}'.format(errors[n_rffs[0]] * n_trials / (1+trial_count))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print 'SE kernel \t{} \t{:.4}sec'.format(errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    filename = 'output/timing/%s_exact_gauss_(LibLinear)_svm.pk' % data_name
    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)
    """

def plot_classification_errors(data_name, algo_names, filename = 'classification'):
    plt.figure(figsize = (6,4))
    ylim_ticks = [1,0]
    for algo_name in algo_names:
        try:
            with open('output/%s_%s_svm.pk' % (data_name, algo_name), 'rb+') as f:
                data = pickle.load(f)
                print('Loading output/%s_%s_svm.pk' % (data_name, algo_name))
        except IOError:
            with open('output/timing/%s_%s_svm.pk' % (data_name, algo_name), 'rb+') as f:
                data = pickle.load(f)
                print('Loading output/timing/%s_%s_svm.pk' % (data_name, algo_name))
        # else:
        #     raise IOError('No such file: %s_%s_svm.pk in output(/timing)' % (data_name, algo_name))
        x = filter(lambda k: isinstance(k, numbers.Number), data.keys())
        x.sort()
        means = np.array([np.mean(data[k]) for k in x])

        ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        ylim_ticks[1] = max(ylim_ticks[1], np.max(means))
        plt.plot(x, means, '.-', label = algo_name.replace('_', ' '), linewidth = 1)
    
    plt.xlabel(r'\# random features')
    plt.ylabel(r'Average accuracy')
    plt.yscale('log')

    if data_name == 'adult':
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        yticks_spacing = 1e-2 # space between y ticks
    elif data_name == 'bank':
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
        yticks_spacing = 4e-3
    
    yticks_lim_integer = (1 + int(ylim_ticks[0] / yticks_spacing), int(ylim_ticks[1] / yticks_spacing)) # floor and ceil
    plt.minorticks_off()
    plt.yticks(yticks_spacing * np.arange(yticks_lim_integer[0], 1 + yticks_lim_integer[1]))

    xticks_spacing = 24
    xticks_lim = [(int(min(plt.xticks()[0]) / xticks_spacing) + 1) * xticks_spacing, int(max(plt.xticks()[0]) / xticks_spacing) * xticks_spacing]
    xticks_lim[0] = max(xticks_lim[0], 0)
    plt.xticks(range(xticks_lim[0], xticks_lim[1]+1, xticks_spacing))
    plt.xlim(min(xticks_lim), max(xticks_lim))

    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_%s.eps' % (data_name, filename))
    plt.show()
    plt.clf()

def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    data_name = ['adult', 'bank', 'covtype', 'MNIST'][0]
    if data_name == 'adult':
        X, y = load_adult()
    elif data_name == 'covtype':
        X, y = load_covtype()
    elif data_name == 'bank':
        X, y = load_bank()
    elif data_name == 'MNIST':
        X, y = load_MNIST()
    print(data_name, len(y), np.mean(y))
    if data_name in ['adult', 'bank']:
        X = whiten_data(X)[0]
    np.random.seed(0)

    if data_name == 'adult':
        X_train = X[:32652]
        y_train = y[:32652]
        X_test = X[32652:]
        y_test = y[32652:]
        scale = 6.0
        C = 1.0
    elif data_name == 'bank':
        X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
        scale = 10.0
        C = 4.0
    elif data_name == 'covtype':
        X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
        scale = 20.0
        C = 1.0
    elif data_name == 'MNIST':
        X_train = X[:55000]
        y_train = y[:55000]
        X_test = X[55000:]
        y_test = y[55000:]
        scale = int(0.5*28)
        C = 1.0

    keys = ['iid', 'ort', 'iid_fix_norm', 'ort_fix_norm', 'ort_weighted', 'HD_1', 'HD_3', 'HD_1_fix_norm', 'HD_3_fix_norm']
    keys = ['iid', 'ort', 'HD_1', 'HD_3', 'ort_weighted']
    keys = ['iid', 'ort']
    algos = algos_generator(keys, scale = scale)

    # for algo_name, feature_gen_handle in algos.items():
    #     print algo_name
    #     phi = feature_gen_handle(X_train, 784, 0)
    #     print phi, np.linalg.norm(phi)
    #     y_test_fit1 = svm.fit_from_feature_gen(X_train, y_train, X_test, C, lambda a: feature_gen_handle(a, 784, 0))
    #     print y_test_fit1

    # classification_error_kernel(data_name, X_train, y_train, X_test, y_test, C, scale = scale)
    classification_error_n_rff(data_name, algos, X_train, y_train, X_test, y_test, C)
    # plot_classification_errors(data_name, keys + ['exact_gauss']+['exact_gauss_(LibLinear)'])
    # plot_runtimes(data_name, [key+'_svm' for key in keys], filename = 'classification')

    # for C in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    #     for scale in [1.0, 4.0, 10.0, 20.0, 40.0, 100.0]:
    # for n_rff in [8,12,16,20,24,28,32]:
    #     err = 0.0
    #     for seed in range(20):
    #         y_test_fit1 = svm.fit_from_feature_gen(X_train, y_train, X_test, C, lambda a: kernels.iid_gaussian_RFF(a, n_rff, seed, scale))
    #         y_test_fit2 = svm.fit_from_feature_gen(X_train, y_train, X_test, C, lambda a: kernels.ort_gaussian_RFF(a, n_rff, seed, scale))
    #         print n_rff, np.mean(np.abs(y_test_fit1 != y_test)) - np.mean(np.abs(y_test_fit2 != y_test))
    #     print(C, scale, n_rff, err)
    # y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, C, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    # print(C, scale, np.mean(np.abs(y_test_fit != y_test)))

if __name__ == '__main__':
    main()
