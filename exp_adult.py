import sys, time, csv, pickle, numbers
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import svm
sys.path.append('/home/fc443/project/regression_task/')
import kernels
from exp_uci import split_data, whiten_data, algos_generator

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

def classification_error_n_rff(data_name, algos, X_train, y_train, X_test, y_test, C):
    timing = False
    if timing:
        n_seeds = 1000
    else:
        n_seeds = 50
    
    if data_name == 'adult':
        n_rffs = [4, 8, 12, 16, 20] + range(24, 256 + 1, 24)

    for algo_name, feature_gen_handle in algos.items():
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
        
        with open(filename, 'wb') as f:
            pickle.dump(errors, f)

def classification_error_kernel(data_name, X_train, y_train, X_test, y_test, C, scale = None):
    if data_name == 'adult':
        n_rffs = [4, 244]
    
    errors = {}
    errors['runtimes'] = {}
    n_trials = 10
    start_time = time.clock()
    for _ in range(n_trials):
        y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, C, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[0]] = np.mean(np.abs(y_test_fit != y_test))
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print '{} \t{} \t{:.4}sec'.format('SE kernel', errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    if n_trials > 1:
        filename = 'output/timing/%s_exact_gauss_svm.pk' % data_name
    else:
        filename = 'output/%s_exact_gauss_svm.pk' % data_name

    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)

def plot_classification_errors(data_name, algo_names, filename = 'classification'):
    plt.figure(figsize = (6,4))
    ylim_ticks = [1,0]
    for algo_name in algo_names:
        with open('output/%s_%s_svm.pk' % (data_name, algo_name), 'rb+') as f:
            data = pickle.load(f)
        x = filter(lambda k: isinstance(k, numbers.Number), data.keys())
        x.sort()
        means = np.array([np.mean(data[k]) for k in x])
        ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        ylim_ticks[1] = max(ylim_ticks[1], np.max(means))
        plt.semilogy(x, means, '.-', label = algo_name.replace('_', ' '), linewidth = 1)
    
    plt.xlabel(r'\# random features')
    
    plt.ylabel(r'Average accuracy')

    if data_name == 'adult':
        yticks_spacing = 1e-2 # space between y ticks
    
    yticks_lim_integer = (1 + int(ylim_ticks[0] / yticks_spacing), int(ylim_ticks[1] / yticks_spacing)) # floor and ceil
    plt.minorticks_off()
    plt.yticks(yticks_spacing * np.arange(yticks_lim_integer[0], 1 + yticks_lim_integer[1]))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    xticks_spacing = 24
    xticks_lim = [(int(min(plt.xticks()[0]) / xticks_spacing) + 1) * xticks_spacing, int(max(plt.xticks()[0]) / xticks_spacing) * xticks_spacing]
    xticks_lim[0] = max(xticks_lim[0], 0)
    plt.xticks(range(xticks_lim[0], xticks_lim[1]+1, xticks_spacing))

    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_%s.eps' % (data_name, filename))
    plt.show()
    plt.clf()

def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    data_name = ['adult', 'covtype'][0]
    if data_name == 'adult':
        X, y = load_adult()
    elif data_name == 'covtype':
        X, y = load_covtype()
    print(data_name, len(y), np.mean(y))
    X = whiten_data(X)[0]
    np.random.seed(2)
    X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
    
    if data_name == 'adult':
        scale = 4.0
        C = 1.0
    elif data_name == 'covtype':
        scale = 20.0
        C = 1.0

    keys = ['iid', 'ort', 'iid_fix_norm', 'ort_fix_norm', 'ort_weighted', 'HD_1', 'HD_3', 'HD_1_fix_norm', 'HD_3_fix_norm']
    algos = algos_generator(keys, scale = scale)

    # classification_error_kernel(data_name, X_train, y_train, X_test, y_test, C, scale = scale)
    # classification_error_n_rff(data_name, algos, X_train, y_train, X_test, y_test, C)
    # keys = ['iid', 'ort_weighted', 'ort', 'HD_3_fix_norm']
    # plot_classification_errors(data_name, keys + ['exact_gauss'])

    y_test_fit = svm.fit_from_feature_gen(X_train, y_train, X_test, C, lambda a: kernels.iid_gaussian_RFF(a, 256, 0, scale))
    print(C, scale, np.mean(np.abs(y_test_fit != y_test)))
    # y_test_fit = svm.fit_from_kernel_gen(X_train, y_train, X_test, C, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    # print(C, scale, np.mean(np.abs(y_test_fit != y_test)))

if __name__ == '__main__':
    main()
