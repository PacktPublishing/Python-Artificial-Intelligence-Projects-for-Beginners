# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:59:22 2021

@author: user
"""

import pandas as pd


def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=';')
    len(d)
    
    d['pass'] = d.apply(lambda row: 1 if (row['y']) == 'yes' else 0, axis=1)
    d.head()
    
    # use one-hot encoding on categorical columns
    d = pd.get_dummies(d, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'])
    d.head()
    
    # shuffle rows
    d = d.sample(frac=1)
    # split training and testing data
    d_train = d[:3000]
    d_test = d[3000:]
    #3000 data pertama
    d_train_att = d_train.drop(['pass'], axis=1)
    d_train_pass = d_train['pass']
    #setelah dikurang 3000 data pertama sisanya
    d_test_att = d_test.drop(['pass'], axis=1)
    d_test_pass = d_test['pass']
    #semua data
    d_att = d.drop(['pass'], axis=1)
    d_pass = d['pass']

    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)


    

