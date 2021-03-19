import pandas as pd

def preparation(datasetpath):
    data = pd.read_csv(datasetpath, sep=',')
    # a = len(data)
    # print(a)
    dat = data.sample(frac=1)
    dat_train = dat[:2220]
    dat_test = dat[2220:]

    dat_test_atr = dat_test.drop(['classification'], axis=1)
    dat_test_cls = dat_test['classification']
    dat_train_atr = dat_train.drop(['classification'], axis=1)
    dat_train_cls = dat_train['classification']

    total = [[dat_test_atr,dat_test_cls], [dat_train_atr,dat_train_cls]]
    return total

def training(dat_train_atr,dat_train_cls):
    from sklearn import tree as tr
    trainin = tr.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    trainingg = trainin.fit(dat_train_atr,dat_train_cls)
    return trainingg

def testing(training, testdataframe):
    return training.predict(testdataframe)