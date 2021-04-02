import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def preparation():
    data = pd.read_csv('../Chapter01/dataset/karyawan.txt', sep=',', usecols=[0,1,2,3,4,5], header=None, names=['umur','kendaraan','jarak','menikah','tempatTinggal','produktif'])
    data = data.sample(frac=1)
    data = [data.iloc[:,:5], data.iloc[:, 5:]]

    berkasAttr = data.pop(0)
    bekasLabel = data.pop(0)

    length = int(len(bekasLabel)*0.75)

    trainLabel = bekasLabel[:length]
    trainAttr = berkasAttr[:length]

    testLabel = bekasLabel[length:]
    testAttr = berkasAttr[length:]

    return [[trainAttr, trainLabel], [testAttr, testLabel]]

def training(trainAttr, trainVar):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(trainAttr, trainLabel)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)
