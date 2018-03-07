# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:45:49 2018

@author: mdegra200 adapted from example on kaggle
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#loading the wine data set
df = pd.read_csv('C:/data/wine/wine-reviews/winemag-data-130k-v2.csv')
df.head(5)

#looking at the stats for data, mean score of 88.44, mean price 35.36
df.describe()

y = df['points']

#also removing stop words
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',analyzer='word')
x = vectorizer.fit_transform(df['description']) # x is vectorized description

from sklearn.model_selection import train_test_split
#description x, y is points
#Split arrays or matrices into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)

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





