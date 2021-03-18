# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:14:08 2021

@author: MAJESTY
"""
import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',')
    len(d)
    
    d = d.sample(frac=1)
    
    d_train = d[:5455]
    d_test = d[1360:]
    
    d_train_att = d_train.drop(['Bankrupt?'], axis=1) #fitur
    d_train_pass = d_train['Bankrupt?'] #label
    
    d_test_att = d_test.drop(['Bankrupt?'], axis=1)
    d_test_pass = d_test['Bankrupt?']
    
    d_att = d.drop(['Bankrupt?'], axis=1)
    d_pass = d['Bankrupt?']
    
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)