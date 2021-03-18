import pandas 
from sklearn.model_selection import cross_val_score

def preparation():
    dfs = pandas.read_csv('Chapter01/dataset/Absenteeism_at_work.csv', sep=';')
    dfs = dfs.sample(frac=1)

    dfs_train = dfs[:370]
    dfs_test = dfs[370:]

    dfs_train_attribute = dfs_train.drop(['Disciplinary failure'], axis=1)
    dfs_train_Disciplinary_failure = dfs_train['Disciplinary failure']

    dfs_test_attribute = dfs_test.drop(['Disciplinary failure'], axis=1)
    dfs_test_Disciplinary_failure = dfs_test['Disciplinary failure']

    data = [[dfs_train_attribute,dfs_train_Disciplinary_failure], [dfs_test_attribute, dfs_test_Disciplinary_failure]]
    return data

def training(dfs_train_attribute,dfs_train_Disciplinary_failure):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(dfs_train_attribute, dfs_train_Disciplinary_failure)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)