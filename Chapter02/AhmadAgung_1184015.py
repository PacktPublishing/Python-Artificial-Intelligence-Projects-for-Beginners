# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:58:25 2021

@author: ahmad
"""

import pandas as pd
from sklearn import preprocessing

def preparation(datasetpath):
    f = pd.read_csv(datasetpath, sep=',', header=None, error_bad_lines=False, warn_bad_lines=False, usecols=[0,1,2,3,4], names=['AT','V','AP','RH','PE'])
    
    f = pd.get_dummies(f, columns=['V','AP','RH','PE'])
    
    encode = preprocessing.LabelEncoder()
    f['AT'] = encode.fit_transform(f['AT'])
    
    f= f.sample(frac=1)
    
    f_att = f.iloc[:, 1:118]
    f_label = f.iloc[:, 0:1]
    
    f_train_att = f_att[:6000]
    f_train_label = f_label[:6000]
    f_test_att = f_att[6000:]
    f_test_label = f_label[6000:]
    
    f_train_label = f_train_label['GAS']
    f_test_label = f_test_label['GAS']
    
    print(f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label)

def training(f_train_att, f_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)
    clf = clf.fit(f_train_att, f_train_label)
    return clf


def testing(clf, f_test_att):
    return clf.predict(f_test_att.head())
