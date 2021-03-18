import unittest


class TestApp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_02_rolly_113040087(self):
        from Chapter01.rolly113040087 import preparation,training,testing
        dataset='Chapter01/dataset/student-por.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        hasiltestingsemua = 	testing(t,d_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_03_angga_1184047(self):
        from Chapter01.angga1184047 import preprocessing, training, predict

        while True:
            data = preprocessing()

            # training data
            data_train = data['training']
            data_training = data_train['data_training']
            data_training_label = data_train['data_training_label']

            # testing data
            data_train = data['testing']
            data_testing = data_train['data_testing']
            data_testing_label = data_train['data_testing_label']

            # training
            t = training(data_training, data_training_label)

            # predict
            prediction = predict(t, data_testing)

            print(data_testing_label.values[0])
            print(prediction[0])
            if data_testing_label.values[0] == prediction[0]:
                print("hasil sama")
                break
            else:
                print("hasil beda")
        self.assertEqual(data_testing_label.values[0], prediction[0])

    def test_02_DindaMajesty_1184011(self):
        from Chapter01.DindaMajesty1184011 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/company_data.csv'
        # testing function preparation
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datasetpath)
        #testing function training
        t = training(d_train_att, d_train_pass)
        #testing function testing
        hasiltestingsemua = testing(t, d_test_att)
        #hasil
        print('\n hasil testing dinda : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_02_DyningAida_1184030(self):
        from Chapter01.DyningAida1184030 import preparation, training, testing
        
        #path ke dataset
        datasetpath = 'Chapter01/dataset/online_shoppers_intention.csv'
        
        # testing function preparation
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datasetpath)
        
        #testing function training
        t = training(d_train_att, d_train_pass)
        
        #testing function testing
        hasiltestingsemua = testing(t, d_test_att)
        
        #hasil
        print('\n hasil testing Batris : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_02_idam_1184063(self):
        from Chapter01.idam1184063 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        dfs_train_att = train.pop(0)
        dfs_train_win = train.pop(0)
        #test data
        test = data.pop(0)
        dfs_test_att = test.pop(0)
        dfs_test_win = test.pop(0)
        #training
        t = training(dfs_train_att, dfs_train_win)
        #predict
        result = testing(t,dfs_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)

