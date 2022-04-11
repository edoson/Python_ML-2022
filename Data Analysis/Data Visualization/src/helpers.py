import pandas as pd

def calc_requests_proportion_by_dimension(df, dim_col):
    counts_by_dim = df.groupby(dim_col).size().rename('request_counts')
    props_by_dim = counts_by_dim.div(len(df)).rename('reqeust_proportions')
    return pd.concat([counts_by_dim, props_by_dim], axis=1)


def calc_success_rate(grp):
    success_rate = len(grp[grp.is_success == 'yes']) / len(grp)
    return pd.Series({'success_rate' : success_rate,
                      'success_rate_ci': success_rate / 20})