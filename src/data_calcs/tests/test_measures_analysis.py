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
                             
    
# ------------------------- Class FakeClassifier -------------    
class FakeClassifier:
    pass 


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()