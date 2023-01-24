# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures


def get_outliers(df):
    
    '''takes  a dataframe and graph boxplot to show outliers'''
    
    for col in df.columns.tolist():
        sns.boxplot(x=col, data=df)
        plt.show()

def get_barplot_county(df):
    
    '''takes a dataframe and graph a barplot of home value and county'''
    
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x='county', y='tax_amount', data=df)
    avg_tax_amount = df.tax_amount.mean()
    plt.axhline(avg_tax_amount, label="Avg_house_value = 4.22K", color='yellow')
    plt.legend()
    plt.yticks(ticks = [0,100000,200000,300000,400000,500000,600000], labels=['0', '100K ', '200K', '300K', '400K', '500K', '600K'])
    plt.ylabel('Home  Value  ($)')
    plt.title('House Value across counties')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
 
    
def get_barplot_bathroom_bedroom(df):
    
    '''takes a dataframe and graph two barplot of bathroom vs home value and bedroom vs home value '''
    
    plt.figure(figsize=(8,10))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=1,
                    wspace=0.4,
                    hspace=0.4)
    
    plt.subplot(211)
    ax = sns.barplot(x='tax_amount',y='bathroom', data=df, color='steelblue', orient='h' )
    median = df.bathroom.median()
    plt.annotate(f' Median bathroom: {median}',(200000,2))
    plt.xticks(ticks = [0,100000,200000,300000,400000,500000, 600000,700000, 800000], labels=['0', '100K ', '200K', '300K', '400K', '500K', '600K','700K', '800K'])
    plt.xlabel('Home  Value  ($)')
    plt.ylabel('Number of bathroom')
    plt.ylim(-1,5)
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Home value with number of bathroom")
    
    plt.subplot(212)
    ax = sns.barplot(x='tax_amount',y='bedroom', data=df, color='skyblue', orient='h' )
    median = df.bedroom.median()
    plt.annotate(f' Median bedroom: {median}',(100000,1))
    plt.xticks(ticks = [0,100000,200000,300000,400000,500000, 600000,700000, 800000], labels=['0', '100K ', '200K', '300K', '400K', '500K', '600K','700K', '800K'])
    plt.xlabel('Home  Value  ($)')
    plt.ylabel('Number of bedroom')
    plt.ylim(-1,5)
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Home value with number of bedroom")
    plt.show()
    
    
def get_barplot_one_bathroom_two_bedroom(df):
    
    '''takes a dataframe and graph a barplot of one bathroom vs home value and two bedroom vs home value '''

    plt.figure(figsize=(15,10))
    plt.subplot(121)
    ax = sns.barplot(x='bathroom', y='tax_amount',data=df[df.bathroom==1])
    plt.yticks(ticks = [0,50000,100000,150000,200000,250000,300000,350000], labels=['0', '50K','100K ', '150K','200K','250K' ,'300K','35K'])
    plt.ylabel('Home  Value  ($)')
    plt.xlabel('Number of Bathroom')
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Home value with number of bathroom")
    
    plt.subplot(122)
    ax = sns.barplot(x='bedroom', y='tax_amount', data=df[df.bedroom==2])
#     sns.barplot(x='bedroom', y='tax_amount', data=df,)
    plt.yticks(ticks = [0,50000,100000,150000,200000,250000,300000,350000], labels=['0', '50K','100K ', '150K','200K','250K' ,'300K','35K'])
    plt.ylabel('Home  Value  ($)')
    plt.xlabel('Number of Bedroom')
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Home value with number of bedroom")
    plt.show()
    
    
def get_lmplot(df):
    
    '''takes a dataframe and plot a graph of sqft vs home value '''

    plt.figure(figsize=(10,6))
    ax = sns.lmplot(x='sqft',y='tax_amount', data=df.sample(1000),line_kws={'color':'red'})
    plt.xlabel('Area of Home in sqft')
    plt.ylabel('Home  Value  ($)')
    plt.title("Relation of Area of Home with Home Value")
    plt.ticklabel_format(style='plain')
    plt.show()
    

def get_corelation_map(df):
    
    '''takes a dataframe and graph  heatmap'''
    
    corr = df.corr(method='spearman')
    sns.heatmap(corr, cmap='Purples', annot=True, mask= np.triu(corr))

    
def ind_t_test_one_bathroom_two_bedroom(train):
    
    '''takes in dataframe and runs independent t-test(1_tail,greater than) to compare mean between dataframe attributes'''
    
    # set alpha value to 0.05
    alpha = 0.05
       
    # create a dataframe using boolean masks
    house_with_one_bathroom = train[train.bathroom == 1].tax_amount
    house_with_two_bedroom = train[train.bedroom == 2].tax_amount

     # set null and alternative hypothesis
    null_hypothesis = 'Home with one bathroom has less than or equal value to home with two bedroom' 
    alternative_hypothesis = 'Home with one bathroom has greater value than home with two bedroom'
    
    # print Null Hypothesis followed by a new line
    print(f'Null Hypothesis: {null_hypothesis}\n')

    # print Alternative Hypothesis followed by a new line
    print(f'Alternative Hypothesis: {alternative_hypothesis}\n')

    # verify assumptions:
        # - independent samples
        # - more than 30 observation
        # -equal Variances

    if house_with_one_bathroom.var() != house_with_two_bedroom .var(): 

        # run independent t-test without equl variances
        t, p = stats.ttest_ind(house_with_one_bathroom, house_with_two_bedroom, equal_var = False)
        
        # print alpha value
        print(f'alpha: 0.05')
        
        # print t-statistic value
        print(f't: {t}')

        # print p-value followed by a new line
        print(f'p: {p}\n')

        if t > 0 and p/2 < alpha:
            print('we reject null hypothesis\n')
            print(alternative_hypothesis)
        else:
            print('We fail to reject null hypothesis\n')
            print(f'It appears that {null_hypothesis}')
    else: 
        # run independent t-test with equal variances
        t, p = stats.ttest_ind(house_with_one_bathroom, house_with_two_bedroom)

        # print t-statistic value
        print(f't: {t}')

        # print p-value followed by a new line
        print(f'p: {p}\n')

        if t > 0 and p/2 < alpha:
            print('we reject Null Hypothesis\n')
            print(alternative_hypothesis)
        else:
            print('We fail to reject Null Hypothesis\n')
            print(f'It appears that {null_hypothesis}')
    print('--------------------------------------------------------------------------------------------\n')
    
    
def corelation_sqft_tax_amount( train):
    
    '''takes a dataframe and runs pearsonr test to compare relationship of attributes of a datframe '''
    
    # set alpha value to 0.05
    alpha = 0.05
          
    # set null and alternative hypothesis 
    null_hypothesis = 'There is no linear colrealtion between area and home value'
    alternative_hypothesis = 'There is linear colrealtion between area and home value'

    # run person's correlation test
    r, p = stats.pearsonr(train.sqft, train.tax_amount)

    # print Null Hypothesis followed by a new line
    print(f'Null Hypothesis: {null_hypothesis}\n')

    # print Alternative Hypothesis followed by a new line
    print(f'Alternative Hypothesis: {alternative_hypothesis}\n')

    # print the chi2 value
    print(f'r = {r}') 

    # print the p-value followed by a new line
    print(f'p     = {p}\n')

    if p < alpha:
        print(f'We reject null hypothesis')
        print(alternative_hypothesis)
    else:
        print(f'We fail to reject null hypothesis')
        print(f'There appears to be no significant linear corelation between area and home value')
    print('--------------------------------------------------------------------------------------------\n')

