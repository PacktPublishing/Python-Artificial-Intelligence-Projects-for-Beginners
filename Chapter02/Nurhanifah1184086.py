# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:47:09 2021

@author: User
"""
# In[]
import pandas as pd

# In[]
def preparation(datasetpath):
# In[] 
    d = pd.read_csv('Chapter01/dataset/Callt.txt',
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3],
                    names=['Flow ID', 'Date', 'Time', 'Count'])
# In[]
    d = pd.get_dummies(d, columns=['Date', 'Time',])
# In[]  
     
    d['result'] = d.apply(lambda row: 1 if (row['Count']) == 0 else 0, axis=1)
    d.head()
# In[]  
    d_shuffle = d.sample(frac=1)
    # memetakan atribut dan label
    df_att = d_shuffle.iloc[:, :155]
    df_label = d_shuffle.iloc[:, 155:]

    percent_training = int(len(d)*0.75)
    # data train
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    # data test
    df_test_att = df_att[percent_training:]
    df_test_label = df_label[percent_training:]

    df_train_label = df_train_label['result']
    df_test_label = df_test_label['result']
# In[]
    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label
# In[]
def training(df_train_att, df_train_label):
# In[]    
    from sklearn.ensemble import RandomForestClassifier
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max atribut yang digunakan ialah 4 kolom di setiap independent treenya
    clf = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    # klasifikasi data training df_train_att dan df_train_label
    clf = clf.fit(df_train_att, df_train_label)
# In[]
    return clf
# In[]
def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())
