# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:14:08 2021

@author: MAJESTY
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preparation(datasetpath):
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                    names=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                            'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                            'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                            'spore-print-color', 'population', 'habitat'])

    d = pd.get_dummies(d, columns=['cap-shape','cap-surface','cap-color',
                                   'bruises','odor','gill-attachment','gill-spacing',
                                   'gill-size','gill-color','stalk-shape','stalk-root',
                                   'stalk-surface-above-ring','stalk-surface-below-ring',
                                   'stalk-color-above-ring','stalk-color-below-ring','veil-type',
                                   'veil-color','ring-number','ring-type','spore-print-color',
                                   'population','habitat'])

    encode = LabelEncoder()
    d['class'] = encode.fit_transform(d['class'])

    d = d.sample(frac=1)

    df_att = d.iloc[:, 1:118]
    df_label = d.iloc[:, 0:1]

    df_train_att = df_att[:6400]
    df_train_label = df_label[:6400]
    df_test_att = df_att[6400:]
    df_test_label = df_label[6400:]

    df_train_label = df_train_label['class']
    df_test_label = df_test_label['class']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)
    clf = clf.fit(df_train_att, df_train_label)
    return clf


def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())