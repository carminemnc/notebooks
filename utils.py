import pandas as pd,numpy as np,math,datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
import redshift_connector
from fitter import Fitter, get_common_distributions, get_distributions

sns.set()

'''###################'''
'''Utils & Cleaning'''
'''###################'''

def dfit(data):
    
    """
    This function use the fitter package to fit and find best distributions for given data.
       
    """
    
    f = Fitter(data, distributions=get_common_distributions())
    f.fit()
    print(f.summary())
    print(f.get_best(method='bic'))
    return

def dictextract(lst,v):
    return list(list(zip(*lst))[v])

def time_series(df, conv_dict):

    type_dict = dict(zip(
        list(conv_dict.keys()),
        dictextract(list(conv_dict.values()), 0)
    ))
    names_dict = dict(zip(
        list(conv_dict.keys()),
        dictextract(list(conv_dict.values()), 1)
    ))
    cdf = df.astype(type_dict).rename(columns=names_dict)[
        list(names_dict.values())].fillna(0)

    return cdf

def remove_outliers(data, colname):
    
    """
    This function calculate pmf and cdf functions given a k value and l value,then plot them.

    k: K value of poisson distribution
    l: Lambda value of poisson distribution
    colorline: color of the plot line (default:black)
    
    """
    
    q25 = data[colname].quantile(0.25)
    q75 = data[colname].quantile(0.75)
    iqr = q75-q25
    cut_off = iqr*1.5
    lower, upper = q25-cut_off, q75+cut_off
    outliers = data[(data[colname] < lower) | (data[colname] > upper)]
    outliers_removed = data[(data[colname] > lower) & (data[colname] < upper)]

    print(f'# of outliers:{len(outliers)}')
    print(f'# of records without outliers:{len(outliers_removed)}')
    outliers_removed = outliers_removed.reset_index().drop(columns='index')

    return outliers_removed


'''###################'''
'''Models & Statistics'''
'''###################'''


def poissonDistribution(k,l,colorline ='black'):
    """
    This function calculate pmf and cdf functions given a k value and l value,then plot them.

    k: K value of poisson distribution
    l: Lambda value of poisson distribution
    colorline: color of the plot line (default:black)
    
    """
    
    if l>k:
        print('Lambda value cannot be greater or equal to k')
        return
    
    '''Poisson distribution'''
    
    x = {}
    for i in range(k):
        x[i] = math.pow(2.71828,-1*l) * math.pow(l,i)/math.factorial(i)
    
    pmf =  pd.DataFrame(x.items(), columns=['x','p_x'])
    
    # lambda on pmf distribution
    x_l = pmf[pmf['x']==l]['x']
    y_l = pmf[pmf['x']==l]['p_x']
    
    
    '''CDF (Cumulative distribution function)'''
    def cudf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1,n+1) / n
        
        cdf = pd.DataFrame(
            {
                'x':x,
                'y':y
            }
        )
        return cdf
    
    cdf = cudf(np.random.poisson(l,10000))
    
    print(f'The probability that the event "{l}" occurs is: {round(float(y_l)*100,2)} %')
    
    '''Plot PMF and CDF'''
    
    plt.rcParams.update({"font.family": "Amazon Ember"})
    
    sns.despine()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(pmf['x'],pmf['p_x'], marker='.', color = colorline, linestyle='solid')
    axes[0].plot(x_l,y_l, marker='o', color = 'red', linestyle='solid')
    axes[0].set(xlabel="Number of events",ylabel='P(X <= n)')
    axes[0].set_title('Probability mass function')
    
    axes[1].plot(cdf['x'],cdf['y'], marker='.', color = colorline, linestyle='solid')
    axes[1].set(xlabel="Number of events",ylabel='P(X <= n)')
    axes[1].set_title('Cumulative distribution function')
    
    fig.tight_layout()

    
    return pmf,cdf


