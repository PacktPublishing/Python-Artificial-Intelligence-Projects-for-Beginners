# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:58:49 2021

@author: DyningAida
"""

import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',')
    len(d)
    
    # generate binary label for attribute 'Revenue'
    d['rev'] = d.apply(lambda row: 1 if (row['Revenue']) == 'TRUE' else 0, axis=1)
    d.head()
    
    #shuffle data
    d_shuffle = d.sample(frac=1)
        
    # data training 80%
    percent_training = int(len(d)*0.80)
    d_train = d_shuffle[:percent_training]
    d_train_pass = d_train['rev'] #label
    d_train_att = d_train.drop(['rev'], axis=1) #fitur
        
    # data testing 20%
    d_test = d_shuffle[percent_training:]
    d_test_pass = d_test['rev']
    d_test_att = d_test.drop(['rev'], axis=1)
        
        
    d_att = d_shuffle.drop(['rev'], axis=1)
    d_pass = d_shuffle['rev']
    
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)