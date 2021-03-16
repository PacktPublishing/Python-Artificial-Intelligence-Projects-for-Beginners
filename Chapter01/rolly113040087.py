import pandas as pd


def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=';')
    len(d)
    # generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
    d['pass'] = d.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
    d = d.drop(['G1', 'G2', 'G3'], axis=1)
    d.head()
    # use one-hot encoding on categorical columns
    d = pd.get_dummies(d, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                                   'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                                   'nursery', 'higher', 'internet', 'romantic'])
    d.head()
    # shuffle rows
    d = d.sample(frac=1)
    # split training and testing data
    d_train = d[:500]
    d_test = d[500:]
    #500 data pertama
    d_train_att = d_train.drop(['pass'], axis=1)
    d_train_pass = d_train['pass']
    #setelah dikurang 500 data pertama sisanya
    d_test_att = d_test.drop(['pass'], axis=1)
    d_test_pass = d_test['pass']
    #semua data
    d_att = d.drop(['pass'], axis=1)
    d_pass = d['pass']

    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)

def main
    dataset='Chapter01/dataset/student-por.csv'
    d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
    t = training(d_train_att,d_train_pass)
    

