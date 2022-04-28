import pandas as pd

def compute_acc_stats(grp, col):
    """
    This function recieves a dataframe and an accuracy column, and computes the mean, standard deviation and coeffient of variance (variance per unit of mean)
    """
    m, std = grp[col].mean(),  grp[col].std()
    return m, std, std/m if m > 0 else 0
    
def apply_compute_acc_stats(grp, train_metric_col, test_metric_col):
    """
    Given a dataframe, this function returns a series with mean, std and coeffient of variance stats on training and testing columns
    """
    mean_test_acc, std_test_acc, cv_test_acc = compute_acc_stats(grp, test_metric_col)
    mean_train_acc, std_train_acc, cv_train_acc = compute_acc_stats(grp, train_metric_col)
    return pd.Series({'test_acc_mean': mean_test_acc, 
                      'test_acc_std': std_test_acc,
                      'test_acc_cv': std_test_acc,
                      'train_acc_mean': mean_train_acc, 
                      'train_acc_std': std_train_acc,
                      'train_acc_cv': cv_train_acc,
                      'test_train_diff': mean_train_acc - mean_test_acc,
                      'test_train_diff_percent': 100 * (mean_train_acc - mean_test_acc) / mean_train_acc
                       })