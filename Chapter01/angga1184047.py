from sklearn import tree
from sklearn.model_selection import cross_val_score

import pandas


def preprocessing(data_file_path='Chapter01/dataset/1184047_mobile_phone_price.csv'):
    data_frame_source = pandas.read_csv(data_file_path, sep=',')

    # shuffle (kocok) data
    data_frame_shake = data_frame_source.sample(frac=1)  # frac=1 (means 100% data shuffle)

    # 80% data for training
    eighty_percent = int(len(data_frame_shake) * 0.8)
    data_train = data_frame_shake[:eighty_percent]
    data_train_label = data_train['price_range']
    data_train = data_train.drop(['price_range'], axis=1)

    # 20% data for testing
    data_test = data_frame_shake[eighty_percent:]
    data_test_label = data_test['price_range']
    data_test = data_test.drop(['price_range'], axis=1)

    data = {
        "training": {
            "data_training": data_train,
            "data_training_label": data_train_label
        },
        "testing": {
            "data_testing": data_test,
            "data_testing_label": data_test_label
        }
    }

    return data


def training(data_training, data_training_label):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(data_training, data_training_label)
    scores = cross_val_score(t, data_training, data_training_label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return t


def predict(t, data_testing):
    return t.predict(data_testing)
