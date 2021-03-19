# In[]
#import library
import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree

#data Preprocesing
def preparation(dataset):
    #get dataset
    d = pd.read_csv(dataset)
    d = d.drop(['id'], axis=1) #drop id
    d.head()
    d = pd.get_dummies(d, columns=['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']) #buat data dummies
    
    #normalisasi data menggunakan sklearn StandardScaler()
    std=StandardScaler() 
    columns = ['avg_glucose_level','bmi','age']
    scaled = std.fit_transform(d[['avg_glucose_level','bmi','age']])
    scaled = pd.DataFrame(scaled,columns=columns)
    d=d.drop(columns=columns,axis=1)
    d=d.merge(scaled, left_index=True, right_index=True, how = "left")
    
    #check data null
    d.isnull().sum()
    
    #replace data null dengan 0
    d['bmi'] = d['bmi'].fillna(0)
    
    #acak Rows
    d = d.sample(frac=1)
    
    #pembagian data training (80%) dan data test (20%)
    persen_data_train = int(len(d)*0.50)
    persen_data_test = int(len(d)*0.50)
    
    d_train = d[:persen_data_train]
    d_test = d[persen_data_test:]
    
    d_train_att = d_train.drop(['stroke'], axis=1)
    d_train_stroke = d_train['stroke']
    
    d_test_att = d_test.drop(['stroke'], axis=1)
    d_test_stroke = d_test['stroke']
    
    d_att = d.drop(['stroke'], axis=1)
    d_stroke = d['stroke']
    
    return d_train_att, d_train_stroke, d_test_att, d_test_stroke, d_att, d_stroke


def train(d_train_att, d_train_stroke):
    #Pembuatan Decision tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_stroke)
    return t
    
def test(t, testdataframe):
    return t.predict(testdataframe)










