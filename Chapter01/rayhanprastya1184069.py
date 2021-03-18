import pandas as pd

def preparation(datasetpath):
    data = pd.read_csv(datasetpath, sep=',')
    # a = len(data)
    # print(a)
    dat = data.sample(frac=1)
    dat_train = dat[:522]
    dat_test = dat[522:]

    dat_test_atr = dat_test.drop(['charges'], axis=1)
    dat_test_harga = dat_test['charges']
    dat_train_atr = dat_train.drop(['charges'], axis=1)
    dat_train_harga = dat_train['charges']

    total = [[dat_test_atr,dat_test_harga], [dat_train_atr,dat_train_harga]]
    return total

def training(dat_train_atr,dat_train_harga):
    from sklearn import tree as tr
    trainin = tr.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    trainingg = trainin.fit(dat_train_atr,dat_train_harga)
    return trainingg

def testing(training, testdataframe):
    return training.predict(testdataframe)