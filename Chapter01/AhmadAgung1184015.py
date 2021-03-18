# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:32:24 2021

@author: ahmad
"""

import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    vg = pandas.read_csv('Chapter01/dataset/vgsales.csv', sep=',')
    vg = vg.sample(frac=1)
    vg_train = vg[:5147]
    vg_test = vg[5147:]
    vg_train_attribute = vg_train.drop(['gbs'], axis=1)
    vg_train_gbs = vg_train['gbs']
    vg_test_attribute = vg_test.drop(['gbs'], axis=1)
    vg_test_gbs = vg_test['gbs']
    data = [[vg_train_attribute,vg_train_gbs], [vg_test_attribute, vg_test_gbs]]
    return data

def training(vg_train_att, vg_train_gbs):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(vg_train_att,vg_train_gbs)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)
