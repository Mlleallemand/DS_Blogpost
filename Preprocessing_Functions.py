import numpy as np

import pandas as pd

from scipy.stats import shapiro 

import scipy.stats as stats


def create_dummies_col(df_in, col, delim, delete_original):
    """This function creates dummy columns with (0,1) encoding from
       columns that were based on multiple-selection items in a 
       questionnaire that can clearly be delimited by a character.
       
       Inputs:
       ------------------------------------------------------------
       df_in : DataFrame object containing the column to modify

       col : String. Column name to create dummies for. Dummies
              are named "col_columnValue"
       
       delim : Character by which to split the values in column col
       
       delete-original: Boolean. Indicates whether the original 
                           column should be kept or deleted
                           
       Returns:
       ------------------------------------------------------------
       df : DataFrame object containing the new dummy columns
       
    """
    df = df_in.copy()
    # divide values in column by delimiter ';'
    col_vals = df[col].str.split(pat=delim, expand=True)
    # remove NaNs, extract values, flatten and derive unique set of answers
    col_vals = np.unique(col_vals.dropna().values.flatten())
    col_vals_l = np.copy(col_vals)
    # DEBUG print(col_vals)

    # add a prefix so we later know which column it belonged to
    for x in np.arange(0, len(col_vals)):
        col_vals_l[x] = col + '_' + col_vals[x]
    
    # NaN are replaced with False when doing the string comparison. 
    # So no need to filter beforehand.
    new_cols = pd.concat([df[col].str.contains(x, regex=False, case=True, 
                                               na=False) for x in col_vals],
                          1, keys=col_vals_l).astype(int)
    
    # remove the original column from the dataframe
    if delete_original:
        df.drop(columns=[col], inplace=True)
    df = df.join(new_cols)
    return df

def generate_test_values_matrix(data, y, str_X, filter_on=None):
    """
    Generates a test matrix and corresponding labels from an input DataFrame
        
        data : pandas DataFrame containing the values to test
        
        y : string. The variable to predict. y must be a key in data.
        
        str_X : list or array of strings. The names of the factors used for 
                prediction. str_X must be a key in the 'data' dataframe.
                
        filter_on (None) : list of size 1x2 containing the column label to 
                           filter and the values to filter by. Values can be 
                           a sub-list.              
    """
    if filter_on != None:
        # Select the continent for which to compute the statistic
        mask = data[filter_on[0]].isin([*filter_on[1]])
        df = data.loc[mask, [str_X, y, filter_on[0]]].copy()
    else:
        df = data[[str_X, y]].copy()

    # remove rows with NaNs
    df.dropna(axis=0, how='any', inplace=True)

    # generate data and value matrix
    test_values = [] # must use list due to inequal data sizes
    test_labels = [] # to collect labels for post-hoc tests
    

    for name, group in df.groupby(str_X):
        test_values.append(group[y].values)
        test_labels.append(name)
        print('Number of observations for ', name, '= ', 
              group[y].values.shape[0])
        
    return test_labels, test_values, df

def do_np_post_hoc_tests(data, labels, indices, alpha=0.05, 
                         bonferroni=True):
    """Performs a non-parametric, independent-samples post-hoc Mann-Whitney 
       U test for the country pairs listed in countries, corresponding to the 
       indices listed in indices.
       
       Args:
       ---------------------------
        data (list) : data used to calculate the test statistic. 
                      List of lists due to potential inequal sample sizes
        
        labels (list) : list of strings detailing the comparison, 
                        e.g. 'North America vs. Europe'
        
        indices (list) : list of indices used for calculating the test
        
        alpha (int) : alpha level used for the test
        
        bonferroni (boolean) : whether to use Bonferroni correction on alpha 
                               for the number of tests performed.
       
       Returns:
       ---------------------------
        df_print : Dataframe containing test results
    """
    # use Bonferroni correction by dividing alpha by the #post-hoc test
    if bonferroni:
        alpha = (alpha / len(labels))
        print('Alpha value after Bonferroni correction: ', alpha, '\n')
    
    pairs = zip(indices, labels)
    
    df_print = pd.DataFrame(columns=['label', 'Pval', 'U-stat', 
                                     'is_significant'])
    
    # calculate pairwise, non-parametric post-hoc tests
    count = 0
    for idx, label in pairs:
        stat, pval = stats.mannwhitneyu(data[idx[0]], data[idx[1]])
        
#       # print the results, avoiding newline with end=''
#         print(label, ': (p, U) ', round(pval,4), ', ', round(stat,4), end='') 
#         if pval < alpha:
#             print(' is significant')
#         else:
#             print(' is NOT significant')
            
        df_print = df_print.append(
            pd.DataFrame({'label':label, 'Pval':pval, 
                          'U-stat':stat, 'is_significant':(pval<alpha)}, 
                          index = [count])
        )
        count += 1;
    return df_print