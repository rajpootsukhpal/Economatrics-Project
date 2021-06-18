# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:34:53 2019

@author: User
"""

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.feature_selection import f_regression
from statsmodels.stats.anova import anova_lm
import seaborn as sns

sns.set()
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.precision', 2)
%matplotlib notebook
plt.rcParams['figure.figsize'] = 10, 10

#Load dataset into dataframe
df = pd.read_csv("kc_house_data1.csv")
display(df.head())


df.info()

df['floors'].value_counts()
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X.describe()
#Plotting histogram

X.hist(bins=50, figsize=(20, 15))
plt.show()

# For missing values
print(df.isnull().sum())

#Looking for correlations
corr_matrix = df.corr()
corr_matrix['price'].sort_values(ascending=False)

#Stepwise selection
# Initial Iteration
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X = sm.add_constant(X)
mreg = sm.OLS(y, X).fit()
display(mreg.summary())

(F, pval) = f_regression(X, y)
F,pval

index = list()
index

for i in range(1,len(df.columns)+1):
    if max(F[1:len(df.columns)+1]) == F[i] and max(F[1:len(df.columns)+1]) > stats.f.ppf(q = 1-0.05, dfn = 2, dfd = len(df) - (2+1)):
        index.append(i)
        break
    
modeled_X = df.iloc[:,index]
modeled_X.head()

for j in range(len(df.columns)-2):
    #f test for adding column or model significance
    fvalue1 = []
    for i in range(len(df.columns)-1):
        if i+1 not in index:
            index1 = index + [i+1]
            X = df.iloc[:,index1]
            X = sm.add_constant(X)
            mreg = sm.OLS(y, X).fit()
            #display(mreg.summary())
            fvalue1.append(mreg.fvalue)
        else:
            fvalue1.append(0)
    
    for i in range(len(df.columns)-1):
        if max(fvalue1) == fvalue1[i] and max(fvalue1) > stats.f.ppf(q = 1-0.05, dfn = len(index)+1, dfd = len(df) - (len(index)+2)):
            index.append(i+1)
            break
        
    modeled_X = df.iloc[:,index]
    modeled_X.head()
    
    
    #partial f test for removing insignificanct columns
    cols = list(modeled_X.columns)
    X = modeled_X
    X = sm.add_constant(X)
    mreg = sm.OLS(y, X).fit()
    for i in range(len(index)):
        X = modeled_X.drop([cols[i]],axis = 1)
        X = sm.add_constant(X)
        mreg1 = sm.OLS(y, X).fit()
        res = anova_lm(mreg1,mreg)
        if res.F[1] == np.nan or res.F[1]>=stats.f.ppf(q = 0.95, dfn = len(df) - res.df_resid[1], dfd = res.df_resid[1]):
            continue
        elif res.F[1]<stats.f.ppf(q = 0.95, dfn = len(df) - res.df_resid[1], dfd = res.df_resid[1]):
            modeled_X = modeled_X.drop([cols[i]],axis = 1)
# Final model
modeled_X.head()
mreg = sm.OLS(y, modeled_X).fit()
mreg.summary()

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(modeled_X, y, test_size = 0.2, random_state = 42)

#Fitting multiple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting y_test results using model y_test ~ X_test
regressor.predict(X_test)

#Plotting price ~ sqft_living
y = df.iloc[:,0].values.reshape(-1,1)
X = df.iloc[:,3].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,y)


#Visualising the results
plt.scatter(X, y, color = 'red')
plt.plot(X, reg.predict(X), color = 'blue')
plt.title('sqft_living vs price (Training set)')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()
