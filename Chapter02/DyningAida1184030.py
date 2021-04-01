# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:21:35 2021

@author: DyningAida
"""

import pandas as pd

def preparation(dataset):
    d = pd.read_csv(dataset,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                    names=['parents' ,'has_nurs', 'form','children','housing','finance','social','health'])
    
    # one-hot encoding pada semua kolom
    d = pd.get_dummies(d, columns=['parents' ,'has_nurs', 'form','children','housing','finance','social','health'])
    
    # menambahkan kolom result dan generate label, nilai 0 = tidak layak, 1 = dipertimbangkan, 2 = layak, 4 = sangat layak 
    d['result'] = d.apply(lambda row: 0 if (row['parents_usual']+row['has_nurs_proper']+
                                            row['form_complete']+row['children_1']+row['housing_convenient']+
                                            row['finance_convenient']+row['social_nonprob']+
                                            row['health_recommended']) <= 2 else
                          (1 if(row['parents_usual']+
                                     row['has_nurs_proper']+
                                     row['form_complete']+row['children_1']+
                                     row['housing_convenient']+row['finance_convenient']+
                                     row['social_nonprob']+row['health_recommended']) <= 4 else 
                           (2 if(row['parents_usual']+row['has_nurs_proper']+row['form_complete']+
                                   row['children_1']+row['housing_convenient']+row['finance_convenient']+
                                   row['social_nonprob']+row['health_recommended']) <= 6 else 3 )), axis=1)
    # shuffle data
    d_shuffle = d.sample(frac=1)
    # memetakan atribut dan label
    df_att = d_shuffle.iloc[:, :27]
    df_label = d_shuffle.iloc[:, 27:]
    # banyak data yang akan ditraining
    percent_training = int(len(d)*0.75)
    # data train
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    # data test
    df_test_att = df_att[percent_training:]
    df_test_label = df_label[percent_training:]

    df_train_label = df_train_label['result']
    df_test_label = df_test_label['result']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max atribut yang digunakan ialah 4 kolom di setiap independent treenya
    clf = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    # klasifikasi data training df_train_att dan df_train_label
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())