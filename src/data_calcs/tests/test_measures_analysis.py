'''
Created on Jun 3, 2024

@author: paepcke
'''
from data_calcs.measures_analysis import (
    Classification,
    ClassifictionResult)
import json
import numpy as np
import pandas as pd
import random
import unittest


TEST_ALL = True
#TEST_ALL = False


class MeasuresAnalysisTester(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    # -------------------------- Tests --------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__make_custom_class_labels(self):
        
        # Make a fake classifier object, which has just
        # the needed pseudo attributes:
        
        clf = FakeClassifier()
        clf.classes_ = [True, False]
        classification = Classification()
        
        target_labels = {False : 'Nighttime',
                         True  : 'Daytime'}

        label_map = classification._make_custom_class_labels(clf, target_labels)
        
        self.assertEqual(label_map[0], 'Nighttime')
        
        # Multiclass:
        clf.classes_  = ['large', 'medium', 'small']
        target_labels = {'large'   : 'father',
                         'medium'  : 'mother',
                         'small'   : 'child'}
        
        label_map = classification._make_custom_class_labels(clf, target_labels)
        self.assertEqual(label_map[0], 'father')
        self.assertEqual(label_map[1], 'mother')        
        self.assertEqual(label_map[2], 'child')        
        
    #------------------------------------
    # test_ClassficationResult_new_
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_ClassficationResult_new_(self):
        
        tclass1 = pd.Series([0.5, 0.6, 0.7, 1000],
                            index=['precision', 'recall', 'f1_score', 'support'],
                            name='tclass1'
                            )
                                           
        tclass2 = pd.Series([0.05, 0.06, 0.07, 2000],
                            index=['precision', 'recall', 'f1_score', 'support'],
                            name='tclass2'
                            )

        weighted_avg   =  pd.Series([0.05, 0.06, 0.07, 2000],
                                    index=['precision', 'recall', 'f1_score', 'support'],
                                    name='weighted_avg'
                                    )                                    
                       
        macro_avg      =  pd.Series([0.05, 0.06, 0.07, 2000],
                                    index=['precision', 'recall', 'f1_score', 'support'],
                                    name='macro_avg'
                                    )
        
        acc = 0.8
        balanced_acc = 0.7
        classes_ = ['tclass11', 'tclass2']
        class_label_map = {'tclass1' : 'Day',
                           'tclass2' : 'Night'
                           }   
        conf_mat = pd.DataFrame([[0.7,0.1], [0.0,0.1]])                                 
        
        obj = ClassifictionResult.__new__(
            ClassifictionResult,
            y_test=None,
            y_pred=None,
            tclass1=tclass1,
            tclass2=tclass2,
            weighted_avg=weighted_avg,
            macro_avg=macro_avg,
            acc=acc,
            balanced_acc=balanced_acc,
            classes_=classes_,
            class_label_map=class_label_map,
            conf_mat=conf_mat,
            cls_nms=['tclass1', 'tclass2'],
            comment='Means test.'
            )
        self.assertTrue(isinstance(obj, ClassifictionResult))
        pd.testing.assert_series_equal(obj.tclass1, tclass1)
        pd.testing.assert_series_equal(obj.tclass2, tclass2)
        pd.testing.assert_series_equal(obj.weighted_avg, weighted_avg)
        pd.testing.assert_series_equal(obj.macro_avg, macro_avg)
        self.assertEqual(obj.acc, acc)
        self.assertEqual(obj.balanced_acc, balanced_acc)
        self.assertEqual(obj.classes_, classes_)
        pd.testing.assert_frame_equal(obj.conf_mat, conf_mat)
                        
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
        self.assertEqual(type(mean_res.acc), np.float64)
        self.assertEqual(type(mean_res.balanced_acc), np.float64)
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
        
        print(cls_res)

        
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
        
    
# ------------------------- Class FakeClassifier -------------    
class FakeClassifier:
    pass 


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()