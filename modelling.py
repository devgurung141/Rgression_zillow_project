# imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures


def scale_data(train, validate, test, 
               columns_to_scale=['bedroom', 'bathroom', 'sqft', 'tax_amount'],
               return_scaler=False):
    ''' 
    Takes in train, validate, and test data and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # use sacaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(train[columns_to_scale])
    
    # apply the scaler
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    


def X_y_split(train,validate,test, target):
    
    '''
    takes in train,validate,test and a target variable
    returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    '''  

    X_train = train.drop(columns= ['county', 'tax_amount'])
    y_train = train[[target]]

    X_validate = validate.drop(columns= ['county', 'tax_amount'])
    y_validate = validate[[target]]

    X_test = test.drop(columns= ['county', 'tax_amount'])
    y_test = test[[target]]
   
    return X_train, y_train, X_validate , y_validate, X_test, y_test


def rmse_mean(train,validate, target):
    '''takes train, validate and target and return RMSE of train and validate using mean'''
    
    # compute baseline mean of train and validate data
    train['baseline_mean'] = train[target].mean()
    validate['baseline_mean'] = train[target].mean()
    
    # compute RMSE of train and validate data
    train_RMSE_mean  = mean_squared_error(train[target], train['baseline_mean'], squared=False )
    validae_RMSE_mean = mean_squared_error(validate[target], validate['baseline_mean'], squared=False )
    
    return train_RMSE_mean, validae_RMSE_mean


def print_rmse_baseline(train,validate, target):
    '''takes train, validate and target and print RMSE of train and validate'''
    
    # call a function to get RMSE of train and validate data
    train_RMSE_mean, validae_RMSE_mean= rmse_mean(train,validate, target)
    
    print('RMSE using mean: ')
    print(F'train_RMSE: {train_RMSE_mean}')
    print(f'validate RMSE: {validae_RMSE_mean}')

    
def rmse_median(train,validate, target):
    '''takes train, validate and target and return RMSE of train and validate using mean'''
    
    # compute baseline median of train and validate data
    train['baseline_median'] = train[target].median()
    validate['baseline_median'] = train[target].median()
        
    # compute RMSE of train and validate data
    train_RMSE_median  = mean_squared_error(train[target], train['baseline_median'], squared=False )
    validae_RMSE_median = mean_squared_error(validate[target], validate['baseline_median'], squared=False )
    
    print('RMSE using median')
    print(F'train_RMSE: {train_RMSE_median}')
    print(f'validate RMSE: {validae_RMSE_median}')
    
    return train_RMSE_median, validae_RMSE_median


def linear_regression(X_train,y_train,X_validate,y_validate, target):
    '''takes X_train,y_train,X_validate,y_validate, target 
    use linear_regression model and return RMSE'''
    
    # create the model object
    lm = LinearRegression()  
    
    # Fit the model
    lm.fit(X_train, y_train[target]) 
    
    # Predict train
    y_train['prediction_OLS'] = lm.predict(X_train)
    
    # predict validate 
    y_validate['prediction_OLS'] = lm.predict(X_validate) 
    
    # evaluate train RMSE
    rmse_train = round (mean_squared_error(y_train[target], y_train['prediction_OLS'],squared=False ), 2)
    # evaluate validate rmse
    rmse_validate = round (mean_squared_error(y_validate[target], y_validate['prediction_OLS'],squared=False), 2)
    
#     print(F'train_RMSE: {rmse_train}')
#     print(f'validate RMSE: {rmse_validate}')
    
    return rmse_train, rmse_validate  


def lassoLars(X_train,y_train,X_validate,y_validate, target, alpha):
    '''takes X_train,y_train,X_validate,y_validate, target, alpha
    use lassoLars model and return RMSE'''

    # create the model object
    lars = LassoLars(alpha) 
    
    # Fit the model
    lars.fit(X_train, y_train[target])  
    
    # Predict train
    y_train['prediction_lassoLars'] = lars.predict(X_train)
    
    # predict validate 
    y_validate['prediction_lassoLars'] = lars.predict(X_validate) 
    
    # evaluate train RMSE
    rmse_train = round (mean_squared_error(y_train[target], y_train['prediction_lassoLars'],squared=False ), 2)
    # evaluate validate rmse
    rmse_validate = round (mean_squared_error(y_validate[target], y_validate['prediction_lassoLars'],squared=False), 2)
    
#     print(F'train_RMSE: {rmse_train}')
#     print(f'validate RMSE: {rmse_validate}')
    
    return rmse_train, rmse_validate


def tweedie(X_train,y_train,X_validate,y_validate, target, power, alpha):
    '''takes X_train,y_train,X_validate,y_validate, target, power, alpha
    use tweedie model and return RMSE'''
    
    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)
    
    # Fit the model
    glm.fit(X_train, y_train[target])   
    
    # Predict train
    y_train['prediction_GLM'] = glm.predict(X_train)
    
    # predict validate 
    y_validate['prediction_GLM'] = glm.predict(X_validate) 
    
    # evaluate train RMSE
    rmse_train = round (mean_squared_error(y_train[target], y_train['prediction_GLM'],squared=False ), 2)
    # evaluate validate rmse
    rmse_validate = round (mean_squared_error(y_validate[target], y_validate['prediction_GLM'],squared=False), 2)
    
#     print(F'train_RMSE: {rmse_train}')
#     print(f'validate RMSE: {rmse_validate}')
    
    return rmse_train, rmse_validate


def polynomial(X_train,y_train,X_validate,y_validate, target, degree):
    '''takes X_train,y_train,X_validate,y_validate, target
    use polynomial model and return RMSE'''
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree2 = pf.transform(X_validate)
    
    # create the model object
    lm2 = LinearRegression()
    
    # Fit the model
    lm2.fit(X_train_degree2, y_train[target]) 
    
    # Predict train
    y_train['prediction_polynomial'] = lm2.predict(X_train_degree2)
    
    # predict validate 
    y_validate['prediction_polynomial'] = lm2.predict(X_validate_degree2) 
    
    # evaluate train RMSE
    rmse_train = round (mean_squared_error(y_train[target], y_train['prediction_polynomial'],squared=False ), 2)
    # evaluate validate rmse
    rmse_validate = round (mean_squared_error(y_validate[target], y_validate['prediction_polynomial'],squared=False), 2)
    
#     print(F'train_RMSE: {rmse_train}')
#     print(f'validate RMSE: {rmse_validate}')
    
    return rmse_train, rmse_validate


def rmse_models(X_train, y_train, X_validate, y_validate, train, validate, target):
    '''takesX_train, y_train, X_validate, y_validate, train, validate, target
    return dataframe with models and their RMSE values on train and validate data
    '''
    # get RMSE values
    train_rmse_mean, validate_rmse_mean = rmse_mean(train, validate,'tax_amount')
    rmse_lm_train, rmse_lm_validate = linear_regression(X_train,y_train,X_validate,y_validate, 'tax_amount')
    rmse_lars_train, rmse_lars_validate = lassoLars(X_train,y_train,X_validate,y_validate, 'tax_amount', 1)
    rmse_glm_train, rmse_glm_validate = tweedie(X_train,y_train,X_validate,y_validate, target, 1, 0)
    rmse_poly_train, rmse_poly_validate= polynomial(X_train,y_train,X_validate,y_validate,'tax_amount', 3)
    
    # assing index
    index = ['baseline', 'LinearRegreesion', 'LassoLars(alpha=1)', 'TweedieRegreesor(power=1, alpha=0)','Polynomial Regression(degree=3)']
    
    # create a dataframe
    metric_df = pd.DataFrame({'train_RMSE':[train_rmse_mean, rmse_lm_train, rmse_lars_train, rmse_glm_train, rmse_poly_train],
                         'validate_RMSE': [validate_rmse_mean, rmse_lm_validate, rmse_lars_validate, rmse_glm_validate, rmse_poly_validate]},index=index)
    
    return metric_df

def plot_rmse(df ):
    '''takes a dataframe and plot a graph to compare RMSE of models'''
    
    plt.figure(figsize=(8,10))
    ax = df .plot.bar(rot=75)
    ax.spines[['right', 'top']].set_visible(False)
    plt.ylabel('Home  Value  ($)')
    plt.ylim(200000, 290000)
    plt.title("Comparisons of RMSE")
    
    
def polynomial_test(X_train,y_train,X_test,y_test, target, degree):
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree)

    # fit and transform X_train_scaled
    X_train_degree = pf.fit_transform(X_train)

    # transform X_validate_scaled 
    X_test_degree = pf.transform(X_test)
    
    # create the model object
    lm = LinearRegression()
    # Fit the model
    lm.fit(X_train_degree, y_train[target])   

    # predict test 
    y_test['prediction_polynomial'] = lm.predict(X_test_degree)  
    
    # evaluate test rmse
    rmse_test = round (mean_squared_error(y_test[target], y_test['prediction_polynomial'],squared=False), 2)
    r2 = explained_variance_score(y_test[target], y_test['prediction_polynomial'])
    
    print('Using polynomial Regresion on test')
    print(f'RMSE : { rmse_test}')
    print(f'r2 : {r2}')
    
    return  rmse_test
