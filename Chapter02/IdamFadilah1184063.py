import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def preparation():
    ord = OrdinalEncoder()
    data = pd.read_csv('Chapter01/dataset/toys.txt', sep=',', usecols=[1,2,3,4,5], header=None, names=['index', 'City', 'Gender', 'Age', 'Income', 'Illness'])
    data["Gender"] = ord.fit_transform(data[["Gender"]])
    data["City"] = ord.fit_transform(data[["City"]])
    data["Illness"] = ord.fit_transform(data[["Illness"]])
    data = data.sample(frac=1)
    data = [data.iloc[:,:4], data.iloc[:, 4:]]


    dataAttr = data.pop(0)
    dataVar = data.pop(0)
    

    length = int(len(dataVar)*0.75)

    trainVar = dataVar[:length]
    trainAttr = dataAttr[:length]

    testVar = dataVar[length:]
    testAttr = dataAttr[length:]

    return [[trainAttr, trainVar], [testAttr, testVar]]

def training(trainAttr, trainVar):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)




