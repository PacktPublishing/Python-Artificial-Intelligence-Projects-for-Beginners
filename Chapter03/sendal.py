#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:57:54 2019

@author: awangga
"""

# coding: utf-8

# In[1]: panggil panda dan baca data

import pandas as pd
d = pd.read_csv("Youtube01-Psy.csv")
# In[2]: kelompok spam dan bukan spam
spam=d.query('CLASS == 1')
bukanspam=d.query('CLASS == 0')

# In[3]: memangging lib vektorisasi
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# In[4]: memilih kolom CONTENT dari data d untuk di vektorisasi

dvec = vectorizer.fit_transform(d['CONTENT'])

# In[5]: melihat isi vektorisasi di dvec
dvec

# In[6]: melihar isi data ini ga perlu sih
print(d['CONTENT'][349])
#analyze(d['CONTENT'][349])

# In[7]:melihar daftar kata yang di vektorisasi
daptarkata=vectorizer.get_feature_names()


# In[8]: dikocok bang biar acak ga urut berdasarkan waktu lagi

dshuf = d.sample(frac=1)

# In[9]: bagi dua 300 training 50 testing
d_train=dshuf[:300]
d_test=dshuf[300:]

# In[10]: lakukan training dari data training
d_train_att=vectorizer.fit_transform(d_train['CONTENT'])
d_train_att
# In[11]: lakukan ulang pada data testing
d_test_att=vectorizer.transform(d_test['CONTENT'])
d_test_att
# In[12]: pengambilan label spam atau bukan spam

d_train_label=d_train['CLASS']
d_test_label=d_test['CLASS']

# In[13]: kita coba klasifikasi menggunakan RF dengan 80 tree
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=80)


# In[14]: lakukan training RF dari data yang sudah di transform
clf.fit(d_train_att,d_train_label)

# In[15]: cek score nya

clf.score(d_test_att,d_test_label)

# In[16]: buat terlebih dahulu confusion matrix
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
cm=confusion_matrix(d_test_label, pred_labels)

# In[17]: lakukan cross validation dengan 5 split
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,d_train_att,d_train_label,cv=5)

skorrata2=scores.mean()
skoresd=scores.std()
