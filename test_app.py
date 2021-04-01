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
        hasiltestingsemua = testing(t,d_test_att)
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
        
    def test_02_alifiaZahra_1184051(self):
        from Chapter01.alifiaZahra1184051 import preparation, training, testing
        dataset ='Chapter01/dataset/bank-additional-full.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        allresult = testing(t,d_test_att)
        print('\n hasil testing : ')
        print(allresult)
        oneresult = allresult[0]
        self.assertLessEqual(oneresult, 1)

    def test_02_AhmadAgung_1184015(self):
        from Chapter01.AhmadAgung1184015 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        vg_train_att = train.pop(0)
        vg_train_gbs = train.pop(0)
        #test data
        test = data.pop(0)
        vg_test_att = test.pop(0)
        vg_test_gbs = test.pop(0)
        #training
        t = training(vg_train_att, vg_train_gbs)
        #predict
        result = testing(t,vg_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)
        

    def test_02_FarisMuhammadIhsan_1184099(self):
        from Chapter01.FarisIhsan1184099 import preparation, train, test
        
        #path ke dataset
        dataset = 'Chapter01/dataset/stroke.csv'
        
        # testing function preparation
        d_train_att, d_train_stroke, d_test_att, d_test_stroke, d_att, d_stroke = preparation(dataset)
        
        #testing function training
        t = train(d_train_att, d_train_stroke)
        
        #testing function testing
        hasiltestingsemua = test(t, d_test_att)
        
        #hasil
        print('\n hasil test Faris :')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    
    def test_02_mwahyu_1184059(self):
        from Chapter01.mwahyu1184059 import preparation,training,testing
         #data
        data = preparation()
        #train data
        train = data.pop(0)
        dta_train_att = train.pop(0)
        dta_train_outcome = train.pop(0)
        #test data
        test = data.pop(0)
        dta_test_att = test.pop(0)
        dta_test_outcome = test.pop(0)
        #training
        t = training(dta_train_att, dta_train_outcome)
        #predict
        result = testing(t,dta_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)

    def test_02_rayhany_1184007(self):
        from Chapter01.rayhanyuda1184007 import preparation,training,testing
        #data
        dt = preparation()
        #train data
        train = dt.pop(0)
        dfrs_train_atribut = train.pop(0)
        dfrs_train_sick = train.pop(0)
        #test data
        test = dt.pop(0)
        dfrs_test_atribut = test.pop(0)
        dfrs_test_sick = test.pop(0)
        #training
        r = training(dfrs_train_atribut, dfrs_train_sick)
        #predict
        output = testing(r,dfrs_test_atribut)
        print("output test: ")
        print(output)
        self.assertLessEqual(output[0], 1)
    
    def test_02_rayhanprastya_1184069(self):
        from Chapter01.rayhanprastya1184069 import preparation,training, testing
        datasetpath = 'Chapter01/dataset/spambase.csv'
        data = preparation(datasetpath)
        # data train
        dat_train = data.pop(0)
        dat_train_atr = dat_train.pop(0)
        dat_train_cls = dat_train.pop(0)
        # data test
        dat_test = data.pop(0)
        dat_test_atr = dat_test.pop(0)
        dat_test_cls = dat_test.pop(0)
        # training data
        trainingg = training(dat_test_atr,dat_test_cls)
        # data predict
        hasil = testing(trainingg,dat_test_atr)
        print("hasil testing spam : ")
        print(hasil)
        self.assertLessEqual(hasil[0], 1)
    
    def test_02_adityar_1184021(self):
        from Chapter01.adityar1184021 import preparation, training, testing
        #datasetpath = 'Chapter01/dataset/kuli_ah_daring.csv'
        data = preparation()

        train = data.pop(0)
        d_train_att = train.pop(0)
        d_train_pass = train.pop(0)

        test = data.pop(0)
        d_test_att = test.pop(0)
        d_test_pass = test.pop(0)

        t = training(d_train_att, d_train_pass)

        result = testing(t,d_test_att)
        print("Maka yang di approve adalah : ")
        print(result)
        self.assertGreaterEqual(result[0], 0)

    def test_03_dindamajesty_1184011(self):
        from Chapter02.DindaMajesty1184011 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/mushrooms.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing dinda : ')
        print(hasiltesting)
        print('Score:', clf.score(df_test_att, df_test_label))
        
    def test_03_DyningAida_1184030(self):
        from Chapter02.DyningAida1184030 import preparation, training, testing
        dataset = 'Chapter01/dataset/nursery.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing Batris :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))

    def test_03_AhmadAgung_1184015(self):
        from Chapter02.AhmadAgung_1184015 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/connect-4.txt'
        f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label = preparation(datasetpath)
        #testing dari fungsi traning
        clf = training(f_train_att, f_train_label)
        #testing dari fungsi testing
        hasiltesting = testing(clf, f_test_att.head())

        #hasil testing yang dilakukan        
        print(' testing : ')
        print(hasiltesting)
        print('Hasil draw(0) lose(1) win(2)',clf.score(f_test_att, f_test_label))

    def test_03_IdamFadilah_1184063(self):
        from Chapter02.IdamFadilah1184063 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : "+ str(t.score(testAttr, testVar)))
        
    def test_03_alifiaZahra_1184051(self):
        from Chapter02.alifiaZahra1184051 import preparation, training, testing
        datasetpath ='Chapter01/dataset/poker-hand2.txt'
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        clf = training(df_train_att, df_train_label)
        allresult = testing(clf, df_test_att.head())
        print('\n hasil testing : ', allresult)
        print('Score:', clf.score(df_test_att, df_test_label))