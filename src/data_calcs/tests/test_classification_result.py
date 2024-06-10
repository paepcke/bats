'''
Created on Jun 9, 2024

@author: paepcke
'''
from _io import (
    StringIO)
from data_calcs.classification_result import (
    ClassifictionResult)
import json
import numpy as np
import pandas as pd
import random
import re
import unittest


TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):


    def setUp(self):
        pass

    #------------------------------------
    # test_ClassificationResult_init
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_ClassificationResult_init(self):

        classes_ = [0, 1]
        class_label_map = {1 : 'Day',
                           0 : 'Night'
                           }
        y_test = [0, 0, 1, 0]                                    
        y_pred = [0, 1, 1, 0]
        
        clf_res = ClassifictionResult(y_test, y_pred, 
                                      classes_=classes_, 
                                      class_label_map=class_label_map)
        # Should check for actual values, but no time or energy:
        # Just check for results being floats:
        for attr in ['precision', 'recall', 'f1_score', 'support']:
            self.assertEqual(type(clf_res.Night[attr]), np.float64)
            self.assertEqual(type(clf_res.Day[attr]), np.float64)
            self.assertEqual(type(clf_res.weighted_avg[attr]), np.float64)
            self.assertEqual(type(clf_res.macro_avg[attr]), np.float64)
        self.assertEqual(type(clf_res.acc), float)
        self.assertEqual(type(clf_res.balanced_acc), np.float64)
        self.assertTrue(isinstance(clf_res.conf_mat, pd.DataFrame))
    

    #------------------------------------
    # test_ClassificationResult_mean
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_ClassificationResult_mean(self):

        classes_ = [0, 1]
        class_label_map = {1 : 'Day',
                           0 : 'Night'
                           }
        y_test = [0, 0, 1, 0]                                    
        y_pred = [0, 1, 1, 0]
        
        clf_res = ClassifictionResult(y_test, y_pred, 
                                      classes_=classes_, 
                                      class_label_map=class_label_map)
        
        y_test1 = [0, 0, 1, 0]                                    
        y_pred1 = [0, 0, 1, 0]
        clf_res1 = ClassifictionResult(y_test1, y_pred1, 
                                       classes_=classes_, 
                                       class_label_map=class_label_map)
        
        mean_res = ClassifictionResult.mean([clf_res, clf_res1])
        
        # Should check for actual values, but no time or energy:
        # Just check for results being floats:
        for attr in ['precision', 'recall', 'f1_score', 'support']:
            self.assertEqual(type(mean_res.Night[attr]), np.float64)
            self.assertEqual(type(mean_res.Day[attr]), np.float64)
            self.assertEqual(type(mean_res.weighted_avg[attr]), np.float64)
            self.assertEqual(type(mean_res.macro_avg[attr]), np.float64)
        self.assertEqual(type(mean_res.acc), float)
        self.assertEqual(type(mean_res.balanced_acc), float)
        self.assertTrue(isinstance(mean_res.conf_mat, pd.DataFrame))
    
    #------------------------------------
    # test_to_json
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_json(self):
        
        y_test, y_pred = self.make_y_test_and_pred(n_samples=5)
        cls_res = ClassifictionResult(
            y_test, y_pred,
            classes_=[0,1], 
            class_label_map={0 : 'Toxic', 
                             1 : 'Benign'}
            )
        jstr = cls_res.to_json()
        # Some of the numbers will change,
        # but we'll test json loading, the compare
        # the respective keys:
        expected = ('{"Toxic": "{\\"precision\\":0.0,\\"recall\\":0.0,'
                    '\\"f1_score\\":0.0,\\"support\\":1.0}", "Benign": '
                    '"{\\"precision\\":0.8,\\"recall\\":1.0,'
                    '\\"f1_score\\":0.8888888889,\\"support\\":4.0}", '
                    '"weighted_avg": "{\\"precision\\":0.64,\\"recall\\":0.8,'
                    '\\"f1_score\\":0.7111111111,\\"support\\":5.0}", "macro_avg": '
                    '"{\\"precision\\":0.4,\\"recall\\":0.5,\\"f1_score\\":0.4444444444,'
                    '\\"support\\":5.0}", "acc": 0.8, "balanced_acc": 0.5, '
                    '"classes_": [0, 1], "cls_nms": ["Toxic", "Benign"], '
                    '"class_label_map": "{\\"0\\": \\"Toxic\\", \\"1\\": '
                    '\\"Benign\\"}", "conf_mat": "{\\"0\\":{\\"0\\":0.8,\\"1\\":0.2},'
                    '\\"1\\":{\\"0\\":0.0,\\"1\\":0.0}}", "comment": ""}')
        
        exp_dict  = json.loads(expected)
        jstr_dict = json.loads(jstr)

        self.assertEqual(jstr_dict.keys(), exp_dict.keys())

    #------------------------------------
    # test_from_json
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_json(self):
        jstr = ('{"Toxic": "{\\"precision\\":0.0,\\"recall\\":0.0,'
                    '\\"f1_score\\":0.0,\\"support\\":1.0}", "Benign": '
                    '"{\\"precision\\":0.8,\\"recall\\":1.0,'
                    '\\"f1_score\\":0.8888888889,\\"support\\":4.0}", '
                    '"weighted_avg": "{\\"precision\\":0.64,\\"recall\\":0.8,'
                    '\\"f1_score\\":0.7111111111,\\"support\\":5.0}", "macro_avg": '
                    '"{\\"precision\\":0.4,\\"recall\\":0.5,\\"f1_score\\":0.4444444444,'
                    '\\"support\\":5.0}", "acc": 0.8, "balanced_acc": 0.5, '
                    '"classes_": [0, 1], "cls_nms": ["Toxic", "Benign"], '
                    '"class_label_map": "{\\"0\\": \\"Toxic\\", \\"1\\": '
                    '\\"Benign\\"}", "conf_mat": "{\\"0\\":{\\"0\\":0.8,\\"1\\":0.2},'
                    '\\"1\\":{\\"0\\":0.0,\\"1\\":0.0}}", "comment": ""}')
        cls_res = ClassifictionResult.from_json(jstr)
        # This ClassifictionResult should have attributes
        # for all of ClassifictionResult.ALL_ATTRS, plus
        # 'Toxic' and 'Benign'. Ensure that the types are
        # correct:
        
        # Number of instance attrs:
        expected = len(ClassifictionResult.RESULT_ATTRS) + len(['Toxic', 'Benign'])
        self.assertEqual(len(cls_res.__dict__), expected)
        
        # Check all the pd.Series attrs are indeed Series:
        for attr in ['Toxic', 'Benign', 'weighted_avg', 'macro_avg']:
            self.assertTrue(isinstance(getattr(cls_res, attr), pd.Series))
        
        for attr in ['acc', 'balanced_acc']:
            self.assertEqual(type(getattr(cls_res, attr)), float)
            
        self.assertEqual(type(cls_res.cls_nms), list)
        
        self.assertTrue(isinstance(cls_res.conf_mat, pd.DataFrame))

    # ------------------ Utilities ----------------------
    
    def make_y_test_and_pred(self, n_samples=5):
        
        random.seed = 42
        
        # Probability of a positive class in y_test (controls class distribution)
        p_positive = 0.6  # Adjust this value between 0 and 1 for different class ratios
        
        # Generate y_test with a mix of 1s and 0s based on probability
        y_test = [1 if random.random() < p_positive else 0 for _ in range(n_samples)]
        
        # Generate y_pred with some misclassifications (around 20% error rate)
        error_rate = 0.2  # Adjust this value to control the number of misclassifications
        y_pred = [y if random.random() > error_rate else (1 - y) for y in y_test]
        
        return y_test, y_pred
        
    #------------------------------------
    # test_to_csv 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_csv(self):
        clf_res = self.make_cls_res()
        dst = StringIO()
        
        # Convert the ClassifictionResult to a df,
        # while filling the dst buf with csv:
        csv_df = clf_res.to_csv(clf_res, dst)
        
        # csv_df is now:
        #                  value Precision    Recall        F1 NumCases
        # Measure                                                      
        # Daytime             na   0.55253  0.991906  0.697381     5782
        # Nighttime           na   0.99906    0.8955  0.942932    53713
        # macro_avg           na  0.775795  0.943703  0.820157        5
        # weighted_avg        na  0.955661   0.90487  0.919067        5
        # balanced_acc  0.943703        na        na        na       na
        # acc            0.90487        na        na        na       na
        

        generated_csv = dst.getvalue()
        expected = ('Measure,value,Precision,Recall,F1,NumCases\n'
                    'Daytime,na,0.55253,0.991906,0.697381,5782\n'
                    'Nighttime,na,0.99906,0.8955,0.942932,53713\n'
                    'macro_avg,na,0.775795,0.943703,0.820157,5\n'
                    'weighted_avg,na,0.955661,0.90487,0.919067,5\n'
                    'balanced_acc,0.9437030255112951,na,na,na,na\n'
                    'acc,0.9048700290666378,na,na,na,na\n')
        self.assertEqual(generated_csv, expected)
        
        # String buf is now empty, fill another
        # buffer with the generated csv, and see whether
        # we can regenerate the ClassifictionResult:
        pat = re.compile('^[0-9.]*$')
        str2float = lambda val: float(val) if pat.match(val) else val
        buf = StringIO(generated_csv)
        converters = {col_nm : str2float 
                      for col_nm 
                      in range(0,1+len(csv_df.columns))}
        
        recovered_df = pd.read_csv(buf, 
                                   index_col='Measure', 
                                   converters=converters)
        # The recovered CSV has the default index 0,1,2,...,
        # with an additional column being the index
        pd.testing.assert_frame_equal(recovered_df, csv_df)
        
    #------------------------------------
    # test_printf
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_printf(self):
        clf_res = self.make_cls_res()
        dst = StringIO()
        clf_res.printf(clf_res, dst)
        
        output = dst.getvalue()

        expected = \
			('| Measure      | value              | Precision   | Recall   | F1       | NumCases   |\n'
			 '|:-------------|:-------------------|:------------|:---------|:---------|:-----------|\n'
			 '| Daytime      | na                 | 0.55253     | 0.991906 | 0.697381 | 5782       |\n'
			 '| Nighttime    | na                 | 0.99906     | 0.8955   | 0.942932 | 53713      |\n'
			 '| macro_avg    | na                 | 0.775795    | 0.943703 | 0.820157 | 5          |\n'
			 '| weighted_avg | na                 | 0.955661    | 0.90487  | 0.919067 | 5          |\n'
			 '| balanced_acc | 0.9437030255112951 | na          | na       | na       | na         |\n'
			 '| acc          | 0.9048700290666378 | na          | na       | na       | na         |\n')
        self.assertEqual(output, expected)
        
        
# ------------------- Utilities -------------

    def make_cls_res(self):
        kwargs = {}

        nighttime = pd.Series([0.999060,0.895500,0.942932,53712.800000],
                              index=['precision', 'recall', 'f1_score', 'support'],
                              name='Nighttime'
                              ) 
        daytime = pd.Series([0.552530,0.991906,0.697381,5782.400000],
                              index=['precision', 'recall', 'f1_score', 'support'],
                              name='Daytime'
                              ) 
        weighted_avg = pd.Series([0.955661,0.904870,0.919067,5.200000],
                              index=['precision', 'recall', 'f1_score', 'support'],
                              name='weighted_avg'
                              ) 
        macro_avg = pd.Series([0.775795,0.943703,0.820157,5.200000],
                              index=['precision', 'recall', 'f1_score', 'support'],
                              name='macro_avg'
                              )
        kwargs['Nighttime']    = nighttime
        kwargs['Daytime']      = daytime 
        kwargs['weighted_avg'] = weighted_avg
        kwargs['macro_avg']    = macro_avg
        kwargs['acc']          = 0.9048700290666378
        kwargs['balanced_acc'] = 0.9437030255112951
        conf_mat = pd.DataFrame({0 : 5735.6, 1 : 5613.0,
                                 1 : 46.8,   0 : 48099.8},
                                 index=[0,1])
        # Mean of the confusion matrices:
        kwargs['conf_mat'] = conf_mat
        kwargs['comment']  = 'This is a comment'
        kwargs['classes_'] = [1, 0]
        kwargs['cls_nms']  = ['Daytime', 'Nighttime']
        kwargs['class_label_map'] = {1 : 'Daytime', 0: 'Nighttime'}
        
        clf_res = ClassifictionResult.__new__(
            ClassifictionResult,
            y_test=None,
            y_pred=None,
            **kwargs)
        
        return clf_res

# ------------------- Main --------------    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()