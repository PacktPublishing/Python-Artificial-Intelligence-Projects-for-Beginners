import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',')
    len(d)
    
    d['GS'] = d.apply(lambda row: 1 if (row['Global_Sales']) == 'TRUE' else 0, axis=1)
    d.head()

    d_shuffle = d.sample(frac=1)

    percent_training = int(len(d)*0.80)
    d_train = d_shuffle[:percent_training]
    d_train_pass = d_train['GS'] 
    d_train_att = d_train.drop(['GS'], axis=1)       
    d_test = d_shuffle[percent_training:]
    d_test_pass = d_test['GS']
    d_test_att = d_test.drop(['GS'], axis=1)       
    d_att = d_shuffle.drop(['GS'], axis=1)
    d_pass = d_shuffle['GS']
    
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)