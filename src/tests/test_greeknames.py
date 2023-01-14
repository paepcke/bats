'''
Created on May 28, 2021

@author: Nick Gardner
'''
import os
import unittest

from greeknames.greek_name_classifier import GreekNameClassifier

class NameClassifierTester(unittest.TestCase):
    '''
    To know: some of the method names in unittest
    modules like this one are standardized. All 
    are optional, though there should be at least
    one test method:
    
        o setUpClass()
          Classmethod invoked before any tests 
          are run. Use to initialize constants
          you'll need in many tests, or to create
          some resource, like a temporary directory.
        o tearDownClass()
          Classmethod invoked after all tests have run.
          use to clean up resources, such as deleting
          temporary files
        o setUp()
          Called before every test. Use to set up an
          environment that should be fresh before every
          test
        o tearDown()
          Called after each test is run, regardless of
          its outcome. Use to clean up what was created
          in setUp() method

        o Test methods must start with 'test_'. All other
          methods in this file will not be run as part of
          the testing. But write as many as needed for
          utilities that support the test methods.
    '''


    @classmethod
    def setUpClass(cls):
        # Actions to take only once when
        # the tests in this file are run:
        
        # Directory of this unittest script:
        cls.cur_dir   = os.path.dirname(__file__)
        cls.name_file = os.path.join(cls.cur_dir, 'data/tst_names.txt')

    def setUp(self):
        # Actions to take before each of the tests
        # Note that class variables created in 
        #      setUpClass() can be referred to 
        #      in instance methods using 'self':
        self.nm_classifier = GreekNameClassifier(self.name_file, 
                                                 unittesting=True)

    def tearDown(self):
        # Cleanup actions to take after each of the tests
        pass


# -----------------  Tests for Greek Name Classifier --------

    #------------------------------------
    # test_extract_syllables 
    #-------------------

    def test_extract_syllables(self):
        
        # Test a test that should be successful:
        syllables = self.nm_classifier.extract_syllables('labile')
        truth     = ['la', 'bile']
        self.assertListEqual(syllables, truth)
        
        # Test a case that should fail:
        
        with self.assertRaises(ValueError):
            self.nm_classifier.extract_syllables('not in catalog')

    #------------------------------------
    # test_syllabify 
    #-------------------

    def test_syllabify(self):
        
        syll_dict = self.nm_classifier.syllabify(self.name_file)
        truth     = {'labile' : ['la', 'bile'],
                     'facile' : ['fa', 'cile']
                     }
        
        self.assertDictEqual(syll_dict, truth)
        
# ----------------- Main ------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()