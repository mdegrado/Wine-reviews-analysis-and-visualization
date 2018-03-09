# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:45:49 2018

@author: mdegra200 adapted from example on kaggle
Linear model to predict wine scores from wine taster description
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#loading the wine data set
df = pd.read_csv('C:/data/wine/wine-reviews/winemag-data-130k-v2.csv')
df.head(5)


#looking at the stats for data, mean score of 88.44, mean price 35.36
df.describe()


from sklearn import linear_model

#Ridge regression
#inear least squares with l2 regularization.
#This model solves a regression model where the loss function is the linear least squares function and regularization
# is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. 
#This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape [n_samples, n_targets]).
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
reg = linear_model.Ridge(alpha = 0.5, solver = 'sag') #sag performs better on larger data sets

# y is the points, x is the description
y = df['points'] # df points

#y.head(5)

#also removing stop words
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',analyzer='word')
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#Converts a collection of raw documents to a matrix of TF-IDF features.
x = vectorizer.fit_transform(df['description']) # x is vectorized description


#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
#description x, y is points
#Split arrays or matrices into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)
reg.fit(x_train, y_train)

pred = reg.predict(x_test)
pred #show predictions

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
from sklearn.metrics import r2_score
r2_score(y_test, pred)  #68%

#converting to a df, this took a while to learn for me but ultimately was only a small amount of code
np.c_[pred,y_test.values] #showing results of prediction
pd_df_results = pd.DataFrame(np.c_[pred,y_test.values]) #creating data frame
pd_df_results

pd_df_results.boxplot() #box plot
pd_df_results.describe() #statistics on test predictions, actuals
pd_df_results.head(50) #top 50
pd_df_results.to_csv('C:/data/wine/wine-reviews/out_lr2.csv')


"""
#validating and Viz of model with cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()

y1 = y_test

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, x_test, y1, cv=10)

fig, ax = plt.subplots()
ax.scatter(y1, predicted, edgecolors=(0, 0, 0))
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
"""




