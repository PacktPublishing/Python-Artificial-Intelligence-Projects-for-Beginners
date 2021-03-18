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

    def test_02_AhmadAgung_1184015(self):
        from Chapter01.AhmadAgung1184015 import preparation, training, testing
        
        datasetpath = 'Chapter01/dataset/vgsales.csv'

        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datasetpath)

        t = training(d_train_att, d_train_pass)

        alltesting = testing(t, d_test_att)
        
        print('\n testing : ')
        print(alltesting)
        selectonetes = alltesting[0]
        self.assertLessEqual(selectonetes, 1)
