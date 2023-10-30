# utils libraries
import pandas as pd
import numpy as np
import time,datetime,glob,requests,os,json,cdsapi
from fitter import Fitter, get_common_distributions, get_distributions
import geopandas as gpd
import pygeohash as pgh
import xarray as xr

# visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from plotnine import *

# pre-processing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import miceforest as mf # imputation package
from miceforest import * # imputation package

# ml models
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
from mango import Tuner # bayesian optimization package

# settings
sns.set()
from credentials import credentials

class BayesianRegressors:
    
    def extraTreeRegressor(self, x, y):

        # split into train/test (80/20)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)

        # parameters space
        param_space = {
            'max_depth': range(3, 10),
            'min_samples_split': range(int(0.01*x.shape[0]), int(0.1*x.shape[0])),
            'min_samples_leaf': range(int(0.01*x.shape[0]), int(0.1*x.shape[0])),
            'max_features': ["sqrt", "log2", "auto"]
        }

        # objective function on train/validation with cross-validation
        def objective(space):
            results = []
            for hyper_params in space:
                # model
                model = ExtraTreesRegressor(**hyper_params)
                # cross validation score on 5 folds
                result = cross_val_score(
                    model, x_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
                results.append(result)
            return results

        # optimize, maximizing the negative mean squared error
        tuner = Tuner(param_space,
                    objective,
                    dict(num_iteration=80, initial_random=10)
                    )
        optimisation_results = tuner.maximize()

        best_objective = optimisation_results['best_objective']
        best_params = optimisation_results['best_params']

        # results on test-set
        model = ExtraTreesRegressor(**best_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        test_results = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f'Best RMSE on train-set: {best_objective}')
        print(f'RMSE on test-set: {test_results}')
        print(f'Best Parameters: {best_params}')

        return x_train, x_test, y_train, y_test, best_objective, best_params, model
    
class Maestro:
    
    def outliers(data, column_name, output):

        
        q25 = data[column_name].quantile(0.25)
        q75 = data[column_name].quantile(0.75)
        iqr = q75-q25
        cut_off = iqr*1.5
        lower, upper = q25-cut_off, q75+cut_off

        outliers = data[(data[column_name] < lower) |
                        (data[column_name] > upper)]
        r_outliers = data[(data[column_name] > lower) &
                        (data[column_name] < upper)]

        print(f'25th quantile: {q25} \n75h quantile: {q75} \nIQR: {iqr}\nCut-Off Threshold: {cut_off} \
            \nLower Bound: {lower}\nUpper Bound: {upper}\n# of outliers: {len(outliers)}\n% of outliers: {len(outliers)/len(data)}')

        fig, ax = plt.subplots(1, 2)

        sns.boxplot(data[column_name],
                    notch=True,
                    showcaps=False,
                    flierprops={"marker": "o"},
                    boxprops={"facecolor": (.4, .6, .8, .5)},
                    medianprops={"color": "coral"},
                    fliersize=5, ax=ax[0]).set(title='Outliers boxplot')

        sns.boxplot(r_outliers[column_name],
                    notch=True,
                    showcaps=False,
                    flierprops={"marker": "o"},
                    boxprops={"facecolor": (.4, .6, .8, .5)},
                    medianprops={"color": "coral"},
                    fliersize=5, ax=ax[1]).set(title='Cleaned series')
        
        fig.show()

        if output == 'create_feature':
            data[column_name + '_outliers'] = data[column_name].apply(
                lambda x: 'outlier' if (x < lower) | (x > upper) else np.nan)
        elif output == 'replace_with_na':
            data[column_name] = data[column_name].apply(
                lambda x: np.nan if (x < lower) | (x > upper) else x)
        elif output == 'drop_outliers':
            data = data[(data[column_name] > lower) & (data[column_name] < upper)]

        return data
    
class Voyager:
    
    def copernicus_downloader(self,variables,years,months,days,hours,sub_region,download_path,file_name):
        
        c = cdsapi.Client(url=credentials['copernicus_url'],
                          key=credentials['copernicus_key'],
                          progress=True)
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': years,
                'month': months,
                'day': days,
                'time': hours,
                'area': sub_region,
            },
            f'{download_path}/{file_name}.nc')
        
        return
    
    def copernicus_to_dataframe(self,variables,file_path):
        
        # read data
        xrarray_data = xr.open_dataset(file_path)
        # to dataframe
        data = xrarray_data.to_dataframe().reset_index()
        # renaming variables with original names
        vlist = [e for e in data.columns.to_list() if e not in ('latitude','longitude','time')]
        print(vlist)
        data.rename(columns=dict(zip(vlist,variables)),inplace=True)
        # categorizing precipitation_type
        if 'precipitation_type' in data.columns:
            data['precipitation_type'] = round(data['precipitation_type']).astype(np.int64)
        # converting temperature in Celsius degrees (°C)
        if '2m_temperature' in data.columns:
            data['2m_temperature'] = data['2m_temperature'] - 273.15
        # reordering columns
        data = data[['latitude','longitude','time'] + variables]
        
        return data
    
    
    