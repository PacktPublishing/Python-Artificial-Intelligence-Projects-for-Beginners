import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    dataframe = pandas.read_csv('Chapter01/dataset/diabetes_predictions.csv', sep=';')
    dataframe = dataframe.sample(frac=1)
    d_train = dataframe[:384]
    d_test = dataframe[384:]
    d_train_atribut = d_train.drop(['Havediabetes'], axis=1)
    d_train_sick = d_train['Havediabetes']
    d_test_atribut = d_test.drop(['Havediabetes'], axis=1)
    d_test_sick = d_test['Havediabetes']
    data = [[d_train_atribut,d_train_sick], [d_test_atribut, d_test_sick]]
    return data

def training(d_train_atribut, d_train_sick):
    r = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    r = r.fit(d_train_atribut,d_train_sick)
    return r

def testing(r, testdataframe):
    return r.predict(testdataframe)