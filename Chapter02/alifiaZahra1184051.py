# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:37:48 2021

@author: Lenovo
"""

import pandas as pd


def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',', 
                header=None, error_bad_lines=False, 
                warn_bad_lines=False, 
                usecols=[0,1,2,3,4,5,6,7,8,9,10], 
                names=['s1', 'c1', 's2', 'c2', 's3', 'c3', 's4', 'c4', 's5', 'c5', 'class'])
    
    d['predict_win'] = d.apply(lambda row: 1 if (row['class']) >= 3 else 0, axis=1)
    
    # shuffle rows
    d = d.sample(frac=1)
    
    df_att = d.iloc[:, 1:10]
    df_label = d.iloc[:, 9:12]
    
    # split training and testing data
    df_train_att = df_att[:12500]
    df_train_label = df_label[:12500]
    df_test_att = df_att[12500:]
    df_test_label = df_label[12500:]
    
    df_train_label = df_train_label['predict_win']
    df_test_label = df_test_label['predict_win']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_train_label, df_test_label

def training(df_train_att,df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=0.5, random_state=0, n_estimators=100) 
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf,df_test_att):
    return clf.predict(df_test_att.head())


    

