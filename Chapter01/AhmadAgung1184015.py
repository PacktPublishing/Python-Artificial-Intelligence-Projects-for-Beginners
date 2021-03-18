# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:32:24 2021

@author: ahmad
"""

import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',')
    len(d)
    
    # generate binary label for attribute 'Revenue'
    d['ots'] = d.apply(lambda row: 1 if (row['Other_Sales']) == 'TRUE' else 0, axis=1)
    d.head()
    
    #shuffle data
    d_shuffle = d.sample(frac=1)
        
    # data training 80%
    percent_training = int(len(d)*0.80)
    d_train = d_shuffle[:percent_training]
    d_train_pass = d_train['ots'] #label
    d_train_att = d_train.drop(['ots'], axis=1) #fitur
        
    # data testing 20%
    d_test = d_shuffle[percent_training:]
    d_test_pass = d_test['ots']
    d_test_att = d_test.drop(['ots'], axis=1)
        
        
    d_att = d_shuffle.drop(['ots'], axis=1)
    d_pass = d_shuffle['ots']
    
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    s = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    s = s.fit(d_train_att, d_train_pass)
    return s

def testing(s,testdataframe):
    return s.predict(testdataframe)
