'''
Created on Jun 3, 2024

@author: paepcke
'''
from data_calcs.measures_analysis import Classification
import os
import unittest


TEST_ALL = True
#TEST_ALL = False


class MeasuresAnalysisTester(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    # -------------------------- Tests --------------

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
        
        
    # ------------------ Utilities ----------------------
    
class FakeClassifier:
    pass 


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()