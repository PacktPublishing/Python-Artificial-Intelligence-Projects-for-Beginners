import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    d = pandas.read_csv('Chapter01/dataset/kuli_ah_daring.csv', sep=",")
    d = d.sample(frac=1)
    print("ini dari fungsi sini gan : " +  str(len(d)))
    
    d_train = d[:35]
    d_test = d[35:]
    
    d_train_att = d_train.drop(['no'], axis=1) #fitur
    d_train_pass = d_train['no'] #label
    
    d_test_att = d_test.drop(['no'], axis=1)
    d_test_pass = d_test['no']
    
    d_att = d.drop(['no'], axis=1)
    d_pass = d['no']

    data = [[d_train_att, d_train_pass], [d_test_att, d_test_pass]]
    return data
    
def training(d_train_att,d_train_pass):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)