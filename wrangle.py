# imports

import pandas as pd
import numpy as np

import os
from env import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_zillow():
    '''
    This function reads in zillow data from Codeup database using sql querry into a dataframe and returns a dataframe. 
    '''
        
    querry = """
            SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips 
            FROM properties_2017
            JOIN predictions_2017 USING (parcelid)
            WHERE propertylandusetypeid = 261;
            """
    
    filename = "zillow_2017.csv"
    
    # check if a file exits in a local drive
    if os.path.isfile(filename):
        
        #if yes, read data from a file
        df = pd.read_csv(filename)
        
    else:
        # If not, read data from database into a dataframe
        df = pd.read_sql(querry, get_connection('zillow'))

        # Write that dataframe in to disk for later, Called "caching" the data for later.
        df.to_csv(filename, index=False)
     
    #returns dataframe
    return df


def get_clean_zillow(df):
    '''This function takes a dataframe, rename column names, drop Null values, remove outliers and return clean dataframe'''
   
    # rename columns
    df = df.rename(columns= { 'bedroomcnt':'bedroom',
                         'bathroomcnt': 'bathroom',
                         'calculatedfinishedsquarefeet': 'sqft',
                         'taxvaluedollarcnt': 'tax_amount',
                         'fips':'county'
                        })
    
    # drop Null values
    df = df.dropna()
    
    # change data types to int
    df = df.astype({'bedroom': int, 'bathroom': int})
    
    # sort values 
    df.sort_values(['bedroom','bathroom'],inplace=True)
    
    # remove outliers
    for col in df.columns.tolist():
#         df[col]= df[col][df[col].between(df[col].quantile(.05), df[col].quantile(.95))]
        df[col]= df[col][df[col].between(df[col].quantile(.01), df[col].quantile(.95))]
        
    # drop Null values
    df = df.dropna()
    
    # replace fips with county name 
    df.county = df.county.map({6037.0:'Los Angeles', 6111.0:'Ventura', 6059.0:'Orange'})
    
    # create dummy vairable for categorical variables
    dummies = pd.get_dummies(df['county'])
    
    # concat dummy variables to dataframe
    df = pd.concat([df, dummies],axis=1)
    
    #returns dataframe
    return df


def train_val_test(df, seed=42):
    '''split data into train, validate and test data'''
    
    # split data into 80% train_validate, 20% test
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    # split train_validate data into 70% train, 30% validate
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    
    # returns train, validate, test data
    return train, validate, test


def wrangle_zillow():
    '''This function acquire data, clean data, split data and returns train, vaidate and test data.'''
    
    # acquire zillow data
    df = get_zillow()
     
    # get clean zillow data
    df = get_clean_zillow(df)
    
    # split data into train, validate and test data
    train, validate, test= train_val_test(df, seed=42)
    
    # returns train, validate, test data
    return train, validate, test
