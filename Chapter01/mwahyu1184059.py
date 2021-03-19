import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    dta = pandas.read_csv('Chapter01/dataset/diabetes.csv', sep=',')
    dta = dta.sample(frac=1)
    print(len(dta))
    dta_train = dta[:384]
    dta_test = dta[384:]
    dta_train_attribute = dta_train.drop(['Outcome'], axis=1)
    dta_train_outcome = dta_train['Outcome']
    dta_test_attribute = dta_test.drop(['Outcome'], axis=1)
    dta_test_outcome = dta_test['Outcome']
    data = [[dta_train_attribute,dta_train_outcome], [dta_test_attribute, dta_test_outcome]]
    return data

def training(dta_train_att, dta_train_outcome):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(dta_train_att,dta_train_outcome)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)