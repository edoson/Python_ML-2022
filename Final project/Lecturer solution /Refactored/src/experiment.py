import sys; sys.path.append('../../../Modles and Modeling/src')
from datasets import make_circles_dataframe, make_moons_dataframe
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import os
    
def fit_and_measure(x_train, x_test, y_train, y_test, fit_func, param, measure_func):
    clf = fit_func(x_train, y_train, param)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    return measure_func(y_train, y_train_pred), measure_func(y_test, y_test_pred)

def fit_with_svm(x, y, param):
    clf = SVC(gamma=param)
    clf.fit(x, y)
    return clf

def fit_with_log_reg(x, y, regularization_value):
    clf = LogisticRegression(penalty='l2', C=regularization_value)
    clf.fit(x, y)
    return clf

def prepare_dataset_for_k_fold_cv(dataset, n_samples, k):
    class_a_prop = dataset.label.value_counts(normalize=True)['A']
    class_a_samples = int(n_samples * class_a_prop)
    class_b_samples = n_samples - class_a_samples
    k_fold_ds = dataset.query('label == "A"').head(class_a_samples).append(dataset.query('label == "B"').head(class_b_samples))
    k_fold_ds = k_fold_ds.reset_index(drop=True)
    k_fold_ds['fold'] = k_fold_ds.index.map(lambda x: x % k)
    return k_fold_ds

def prepare_datasets(nls):
    datasets = pd.DataFrame()
    for nl in nls:
        ds1 = make_circles_dataframe(10000, nl)
        ds1['dataset_name'] = 'circles'
        ds1['noise_level'] = nl
        ds2 = make_moons_dataframe(10000, nl)
        ds2['dataset_name'] = 'moons'
        ds2['noise_level'] = nl
        datasets = datasets.append(ds1)
        datasets = datasets.append(ds2)
    return datasets



def run_modeling_experiment(datasets, datasets_type, nls, gamma_range, regularization_values, n_samples, k_folds, clf_types):
    results = []
    for ds_type in datasets_type:
        for nl in nls:
            for n in n_samples:            
                ds = prepare_dataset_for_k_fold_cv(datasets.query('dataset_name == @ds_type and noise_level == @nl'), n_samples=n, k=k_folds)
                print(f'Starting {k_folds}-fold cross validation for {ds_type} datasets with {n} samples and noise level {nl}. Going to train {clf_types} classifiers.')
                for k in range(k_folds):
                    train_ds, test_ds = ds[ds.fold != k], ds[ds.fold == k]
                    assert(train_ds.index.isin(test_ds.index).sum() == 0)

                    x_train, x_test = train_ds[['x','y']], test_ds[['x','y']]
                    y_train, y_test = train_ds['label'].map({'A':1, 'B': -1}), test_ds['label'].map({'A':1, 'B': -1})
                    assert(len(x_train) + len(x_test) == n)
                    assert(len(y_train) + len(y_test) == n)


                    for clf_type in clf_types:
                        if clf_type == 'log_reg':
                            for regularization_value in regularization_values:
                                train_acc, test_acc =  fit_and_measure(x_train, x_test, y_train, y_test, fit_with_log_reg, regularization_value, accuracy_score)
                                results.append((ds_type, n, nl, clf_type, regularization_value, k, train_acc, test_acc, train_acc - test_acc))
                        if clf_type == 'svm':
                            for gamma in gamma_range:
                                train_acc, test_acc = fit_and_measure(x_train, x_test, y_train, y_test, fit_with_svm, gamma, accuracy_score)
                                results.append((ds_type, n, nl, clf_type, gamma, k, train_acc, test_acc, train_acc - test_acc))
    return results