'''
Created on Jun 9, 2024

@author: paepcke
'''

from _functools import (
    reduce)
from sklearn.metrics._classification import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix)
import io
import json
import numpy as np
import pandas as pd
import sys

class ClassifictionResult:
    '''
    Holds all the performance measures of a 
    classification result. Use instances
    as follows. Given an instance <obj>, 
    an some target class names:
    
        obj.<cls_nm>.precision
        obj.<cls_nm>.recall
        obj.<cls_nm>.f1_score
        obj.<cls_nm>.support
        
        obj.macro_avg.precision
        obj.macro_avg.recall
        obj.macro_avg.f1_score
        obj.macro_avg.support
        
        obj.weighted_avg.precision
        obj.weighted_avg.recall
        obj.weighted_avg.f1_score
        obj.weighted_avg.support
        
        obj.acc
        obj.balanced_acc
        
        obj.conf_mat
        
    The balanced accuracy adjusts raw accuracy to 
    compensate for class imbalances and the consequent
    increase of guessing a majority class just by chance,
    rather than by virtue of classifier computations.
    
    The confusion matrix is a dataframe: rows are true values,
    columns are predicted values. The numbers in cells are
    percent of entire population. 
    
    Also available is the class method:
    
       ClassificationResult.mean([classifcation_result1, classifcation_result2, ...])
       
    which returns a new ClassificationResult with the mean
    of all the constituent values across the given objects. 
    '''

    # Every attribute the instances of ClassificationResult
    # will have, except for attributes that correspond to 
    # target class names, such as 'Cat', 'Dog':
    RESULT_ATTRS = [
        'weighted_avg',
        'macro_avg',
        'acc',
        'balanced_acc',
        'classes_',
        'cls_nms',
        'class_label_map',
        'conf_mat',
        'comment'
        ]
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, y_true, y_pred, **kwargs):
        '''
        Allowed kwargs: 
             classes_=None, 
             class_label_map=None, comment='',
             comment=None,
             cm_normalize  ('true', 'pred', 'all', or None)
                           See signature of sklearn.confusion_matrix()
        
        Given the result of a classification as two 
        pd.Series: true labels, y_true, and corresponding
        predicted labels y_pred, store all the values described
        in the class comment.
    
        The optional classes_ argument is a list of class labels
        in the internal order of the classifier that created y_pred.
        For visualization purposes, these labels may be mapped
        to human readable form by the class_label_map. 
        
        The class_label_map is an optional dict mapping raw
        class labels to human readable names: 

            {True  : 'daytime',
             False : 'nighttime}
        or             
            {'sm'  : 'Small'
             'md'  : 'Medium'
             'lg'  : 'Large'}
        
        All access names of the resulting object will be in 
        terms of the human values, as in:
        
             obj.Medium.precision
             obj.daytime.f1_score
                     ...
        
        :param y_true: the true labels, one for each sample
        :type y_true: pd.Series
        :param y_pred: the corresponding predicted values
        :type y_pred: pd.Series
        :parm classes_: list of class labels as used by
            the classifier that created the y_pred
        :type classes_: optional[list[union[int,str]]]
        :param class_label_map: mapping from raw class names
            used by the classifier to human readable names
        :type class_label_map: optional[dict[str : str]]
        '''
        
        truth_len = len(y_true)
        pred_len  = len(y_pred)
        if truth_len != pred_len:
            msg = f"Truth and prediction must have same len, not {truth_len, pred_len}"
            raise ValueError(msg)
        
        # Ensure the y_x are Series:
        if not isinstance(y_true, pd.Series):
            try:
                y_true = pd.Series(y_true)
            except Exception as e:
                raise TypeError(f"Cannot convert y_true to a pd.Series: {e}")
        
        if not isinstance(y_pred, pd.Series):
            try:
                y_pred = pd.Series(y_pred)
            except Exception as e:
                raise TypeError(f"Cannot convert y_pred to a pd.Series: {e}")

        # Unique labels across y_true and y_pred:
        unique_lbs = set(y_true).union(set(y_pred))
        
        # See whether caller passed the internal
        # labels for classes inside the classifier:
        try:
            classes_ = kwargs['classes_']
            # Ensure that none of the classes are 
            # of type np.bool_:
            self.classes_ = [bool(val)
                             for val 
                             in classes_ 
                             if type(val) == np.bool_
                             ]
        except KeyError:
            # Kwarg not provided:
            self.classes_ = None
            
        if self.classes_ is None:
            # No spec, make the labels a default
            # [0...num_classes-1]:
            self.classes_ = list(range(len(unique_lbs)))
            
        try:
            self.class_label_map = kwargs['class_label_map']
        except KeyError:
            # Kwarg not provided
            self.class_label_map = None

        labels, target_names = (list(self.class_label_map.keys()),
                                list(self.class_label_map.values())) \
                                if self.class_label_map is not None \
                                else (self.classes_, self.classes_)
                                
        # Get the following nested dicts:
                                        
        report = classification_report(y_true, y_pred, 
                                       output_dict=True,
                                       labels=labels,
                                       target_names=target_names)
        
        # How does the report name the classes?
        cls_nms = set(report.keys()) - set(['accuracy', 'macro_avg', 'weighted avg', 'macro avg'])
        self.cls_nms = list(cls_nms)
        # We store the values of dicts that the report
        # contains for each for each class as a 
        # Series. This will allow syntax:
        #
        #    <classification_result_obj>.class1.recall
        #    <classification_result_obj>.class2.f1_score
        #                ...
        # 
        # The 'class1', 'class2' will each be an attribute
        # of this instance. These attributes each contain their 
        # Series.
        
        for cls_nm in self.cls_nms:
            value_names = ['precision', 'recall', 'f1_score', 'support']
            nums = pd.Series(report[cls_nm].values(),
                             name=cls_nm,
                             index=value_names
                             )
            setattr(self, cls_nm, nums)
            
        # Same for the 'macro avg' of the result: it also
        # has the elements of the above defined tuple:
        value_names = ['precision', 'recall', 'f1_score', 'support']
        nums = pd.Series(report['macro avg'].values(),
                         name='macro_avg',
                         index=value_names
                         )
        self.macro_avg = nums 
        
        # And same again for the report's 'weighted avg':
        value_names = ['precision', 'recall', 'f1_score', 'support']
        nums = pd.Series(report['weighted avg'].values(),
                         name='weighted_avg',
                         index=value_names
                         )
        self.weighted_avg = nums 
        
        # Finally, the single accuracy number from the report:
        try:
            self.acc = report['accuracy']
        except KeyError:
            msg = (f"No 'accuracy' produced by classification_report(). ",
                   f"Maybe your class_label_map has incorrect labels? ",
                   f"({self.class_label_map})")
            raise ValueError(msg)
        
        # Add the balanced accuracy:
        self.balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # And add the confusion matrix
        # To make the conf matrix use human labels, 
        # need to copy y_true and y_pred, replacing
        # the classifier-internal values there with
        # the ones in class_label_map. First, check
        # that all class labels are covered in the
        # class_label_map:
        
        map_keys = set(self.class_label_map.keys())
        if unique_lbs == map_keys:
            # The class label map covers all of the labels.
            # Make translated y_true and y_pre:
            y_true_mapped = [self.class_label_map[lbl]
                             for lbl 
                             in y_true
                             ]
            y_pred_mapped = [self.class_label_map[lbl]
                             for lbl 
                             in y_pred
                             ]
        else:
            # Retain the internal labels for the conf matrix:
            y_true_mapped = y_true 
            y_pred_mapped = y_pred
            
        # See whether caller specified how the conf mat
        # should be normalized:
        cm_normalization = kwargs.get('cm_normalize', None)
        conf_mat = confusion_matrix(y_true_mapped, y_pred_mapped, 
                                    normalize=cm_normalization
                                    )
        conf_mat_df = pd.DataFrame(conf_mat)
        self.conf_mat = conf_mat_df
        
        # Finally, any comment to describe this result:
        try:
            self.comment = kwargs['comment']
        except KeyError:
            self.comment = ''
        
    #------------------------------------
    # mean
    #-------------------
    
    @staticmethod
    def mean(clf_results, comment=None):
        '''
        Create a new ClassificationResult whose data
        contains the mean of each corresponding data
        item in the given list of ClassificationResult
        instances. I.e. the mean of the weighted_avg
        entries, the balanced_acc entries, etc.
        
        If comment is provided, the new instance's comment
        will be the provided text. Else the comment
        field will be composed of the comments from
        all results in parens after the work 'Mean'
        
        :param clf_results: list of results from which
            to compute the mean
        :type clf_results: list[ClassificationResult]
        :param comment: comment explaining this new
            result
        :type comment: ClassificationResult
        :return new ClassificationResult with all
            the numerical data averaged.
        :rtype ClassificationResult
        '''

        # Combine each result Series (cls1_nm, weighted_avg, ...)
        # into one Series of Series. Sum them as a vectorized
        # operation, and divide by the number of the Series:
        
        # Get the information retrieval type results for each class: 
        # precision, F1, etc. Those are stored Series in each clf_result,
        # in instance variables named the same as the classes,
        # respectively; 
        
        kwargs = {}
        
        # Means of all the target class Series:
        for cls_nm in clf_results[0].cls_nms:
            all_class_ir_results = [getattr(obj, cls_nm)
                                    for obj
                                    in clf_results
                                    ]
            kwargs[cls_nm] = pd.concat(
                all_class_ir_results, axis=1).sum(axis=1) / len(clf_results)
        
        # Get mean of weighted_avg:
        kwargs['weighted_avg'] = pd.concat([obj.weighted_avg
                                            for obj 
                                            in clf_results
                                            ], axis=1).sum(axis=1) / len(clf_results)
                             
        # Same for mean of macro_avg:
        kwargs['macro_avg'] = pd.concat([obj.macro_avg
                                         for obj 
                                         in clf_results
                                         ], axis=1).sum(axis=1) / len(clf_results)

        # The values of the accuracy and balanced accuracy
        # in each object are simple numbers, not Series.
        # So, use np.mean() over simple list of numbers.
        # The 'float(' is to turn the returned np.float64
        # into a Python float to make json.dump(s) work:
        kwargs['acc'] = float(np.mean([obj.acc
                                       for obj 
                                       in clf_results
                                       ]))
        # The 'float(' is to turn the returned np.float64
        # into a Python float to make json.dump(s) work:
        kwargs['balanced_acc'] = float(np.mean([obj.balanced_acc
                                                for obj 
                                                in clf_results
                                                ]))
        
        # Mean of the confusion matrices:
        conf_mat_list = [obj.conf_mat
                         for obj
                         in clf_results
                         ]
        kwargs['conf_mat'] = reduce(lambda df1, df2: df1.add(df2), conf_mat_list) / len(clf_results)
        
        if comment is None:
            # Compose a new comment:
            comments = [f"Comment: {obj.comment}\n"
                        for obj
                        in clf_results
                        ]
            comment = f"Means of results: {comments}"
        
        kwargs['comment']  = comment
        kwargs['classes_'] = list(clf_results[0].classes_)
        kwargs['cls_nms']  = clf_results[0].cls_nms
        kwargs['class_label_map'] = clf_results[0].class_label_map
        
         
        means_obj = ClassifictionResult.__new__(
            ClassifictionResult,
            y_test=None,
            y_pred=None,
            **kwargs)
        
        return means_obj
    
    #------------------------------------
    # to_json
    #-------------------
    
    def to_json(self, path_info=None):
        '''
        Convert this ClassifictionResult into json.
        If path_info is None, return the json string.
        If it is a string, that string is assumed to 
        be the path to a file where the json is to be
        saved. If a file-like, save the json there.
        
        :param path_info: what to do with the resulting json: store or return
        :type path_info: optional[union[str, file-like]]
        :return the json string if path_info is None, else undefined
        :rtype union[str, None]
        '''
        
        # Structure for instance state:
        state_dict = {}
        
        # Get the target-class-specific obj
        # attribute names and corresponding Series
        # values, and add them to the state_dict:
        for cls_nm in self.cls_nms:
            ir_ser = getattr(self, cls_nm)
            state_dict[cls_nm] = ir_ser.to_json()

        state_dict['weighted_avg']     = self.weighted_avg.to_json()      # Series
        state_dict['macro_avg']        = self.macro_avg.to_json()         # Series
        state_dict['acc']              = self.acc
        state_dict['balanced_acc']     = self.balanced_acc
        state_dict['classes_']         = self.classes_
        state_dict['cls_nms']          = self.cls_nms
        state_dict['class_label_map']  = json.dumps(self.class_label_map) # dict
        state_dict['conf_mat']         = self.conf_mat.to_json()  # DataFrame
        state_dict['comment']          = self.comment
        
        if type(path_info) == str:
            try:
                with open(path_info, 'w') as fd:
                    json.dump(state_dict, fd)
                    return
            except Exception as e:
                raise IOError(f"Could not write json to {path_info}: {e}")
        
        elif isinstance(path_info, (io.IOBase, io.TextIOBase)):
            # path-info is an FD:
            json.dump(state_dict, path_info)
            
        else:
            # Return string
            jstr = json.dumps(state_dict)
            return jstr
            
    #------------------------------------
    # from_json 
    #-------------------
    
    @classmethod
    def from_json(cls, jstr):
        
        state_dict = json.loads(jstr)
        # Get an empty ClassifictionResult:
        kwargs = {}
        kwargs['weighted_avg']     = pd.Series(state_dict['weighted_avg'], name='weighted_avg')
        kwargs['macro_avg']        = pd.Series(state_dict['macro_avg'], name='macro_avg')
        kwargs['acc']              = state_dict['acc']
        kwargs['balanced_acc']     = state_dict['balanced_acc']
        kwargs['classes_']         = state_dict['classes_']
        kwargs['cls_nms']          = state_dict['cls_nms']
        # Be tolerant of comment not being present, though
        # it should be, even if as an empty string:
        try:
            kwargs['comment']          = state_dict['comment']
        except KeyError:
            kwargs['comment'] = ''
        class_label_map_raw        = json.loads(state_dict['class_label_map'])
        # Keys of the class_label _map may have been
        # turned into strings from the json decoding.
        # Fix that:
        numeric_keys = cls._numerify_list(list(class_label_map_raw))
        # New dict with the corrected keys:
        numerified_map  = {key : val 
                           for key, val 
                           in zip(numeric_keys, class_label_map_raw.values())
                           }
        
        kwargs['class_label_map'] = numerified_map
        
        # Confusion matrix is a df:
        con_mat_dict   = json.loads(state_dict['conf_mat'])
        con_mat_df_raw = pd.DataFrame.from_dict(con_mat_dict)
        # Fix columns and index to be numbers, if appropriate:
        con_mat_df = cls._numerify_df_indexes(con_mat_df_raw)

        # Next, if there is a class_label_map, turn the 
        # column/index members into human readable form.
        # Use a map function:
        def turn_human(index_el):
            try:
                return numerified_map[index_el]
            except KeyError:
                # No mapping from raw col name to human readable:
                return index_el
        
        new_index = con_mat_df.index.map(turn_human)
        new_cols  = con_mat_df.columns.map(turn_human)
        con_mat_df.columns = new_cols
        con_mat_df.index   = new_index
        
        kwargs['conf_mat'] = con_mat_df
        
        # Still missing are instance attributes that are
        # class names, like 'Cat', or 'Dog'. Find them by
        # set difference:
        saved_attrs = set(state_dict.keys())
        class_nm_attrs = saved_attrs - set(cls.RESULT_ATTRS) 

        for class_nm in class_nm_attrs:
            # The class name attributes are pd.Series with
            # 'precision', 'f1_score', etc.
            kwargs[class_nm] = pd.Series(state_dict[class_nm], name=class_nm)

        cr = ClassifictionResult.__new__(ClassifictionResult, 
                                         y_test=None, y_pred=None,
                                         **kwargs
                                         )
        return cr

    #------------------------------------
    # to_csv
    #-------------------
    
    def to_csv(self, src_info, dest:None):
        '''
        Create first a dataframe, and from it a .csv string
        that is only for the purpose of showing the content
        of this ClassifictionResult instance. The string cannot
        be used to reconstruct a new instance. But the string
        can be printed, or loaded into a spreadsheet.
        
        Arg src_info may be the path to json file that was saved 
        by this class, or it could be ClassificationResult instance.
        
        If dest is None, print the csv to console. Else dest
        is assumed to be a file-like with a write() method. The
        .csv is written there.
        
        Returns the dataframe that from which the .csv was
        constructed.
        
        The df will look like:
	                         value Precision    Recall        F1 NumCases
	        Measure                                                      
	        Daytime             na   0.55253  0.991906  0.697381     5782
	        Nighttime           na   0.99906    0.8955  0.942932    53713
	        macro_avg           na  0.775795  0.943703  0.820157        5
	        weighted_avg        na  0.955661   0.90487  0.919067        5
	        balanced_acc  0.943703        na        na        na       na
	        acc            0.90487        na        na        na       na
	        
                
        :param src_info: either a ClassificationResult or a file path
            to a json string
        :type src_info: union[str, ClassificationResult]
        :param dest: None to print to console, else
            a file-like.
        :type dest: optional[union[file-like]]
        :return dataframe with information about each .csv line
        :rtype pd.DataFrame
        '''
        
        if type(src_info) == str:
            try:
                with open(src_info, 'r') as fd:
                    jstr = fd.read()
                    clf_res_obj = cls.from_json(jstr)
            except Exception as e:
                raise ValueError(f"Could not open {src_info}: {e}")
        else:
            clf_res_obj = src_info

        header = ['value','Precision','Recall','F1','NumCases']
        lines = []
        for cls_nm in clf_res_obj.cls_nms:
            score_ser_nums = getattr(clf_res_obj, cls_nm).copy()
            score_ser = pd.Series(['na'] + list(score_ser_nums.values), 
                                  name=cls_nm,
                                  index = header
                                  )
            # Number of cases (a.k.a. support) may be
            # a float b/c we took the mean across conf matrices:
            score_ser.NumCases = round(score_ser.NumCases) 
            lines.append(score_ser)

        macro_avg_nums = clf_res_obj.macro_avg.copy()
        score_ser = pd.Series(['na'] + list(macro_avg_nums.values), 
                              name='macro_avg',
                              index = header
                              )
        # Number of cases (a.k.a. support) may be
        # a float b/c we took the mean across conf matrices:
        score_ser.NumCases = round(score_ser.NumCases) 
        lines.append(score_ser)
        
        weighted_avg_nums = clf_res_obj.weighted_avg.copy()
        score_ser = pd.Series(['na'] + list(weighted_avg_nums.values), 
                              name='weighted_avg',
                              index = header
                              )
        # Number of cases (a.k.a. support) may be
        # a float b/c we took the mean across conf matrices:
        score_ser.NumCases = round(score_ser.NumCases) 
        lines.append(score_ser)
        
        balanced_acc = pd.Series(['na']*len(header), 
                                 name='balanced_acc', 
                                 index=header)
        balanced_acc.value = clf_res_obj.balanced_acc
        lines.append(balanced_acc)
        
        acc = pd.Series(['na']*len(header), name='acc', index=header)
        acc.value = clf_res_obj.acc
        lines.append(acc)
        
        out_df = pd.DataFrame(lines)
        out_df.index.name = 'Measure'

        out_csv = out_df.to_csv()
        
        if dest is None:
            dest = sys.stdout
        dest.write(out_csv)
        return out_df

    #------------------------------------
    # printf
    #-------------------
    
    def printf(self, src_info, dest:None):
        '''
        Arg src_info may be the path to json file that was saved 
        by this class, or it could be ClassificationResult instance.
        
        :param src_info: either a ClassificationResult or a file path
            to a json string
        :type src_info: union[str, ClassificationResult]
        :param dest: None to print to console, else
            a file-like.
        :type dest: optional[union[file-like]]
        '''
        # Get a df with all the info for each line.
        # Like:
        #                      value Precision    Recall        F1 NumCases
        #     Measure                                                      
        #     Daytime             na   0.55253  0.991906  0.697381     5782
        #     Nighttime           na   0.99906    0.8955  0.942932    53713
        #     macro_avg           na  0.775795  0.943703  0.820157        5
        #     weighted_avg        na  0.955661   0.90487  0.919067        5
        #     balanced_acc  0.943703        na        na        na       na
        #     acc            0.90487        na        na        na       na
    
        # We just want the returned print
        # content df; so ignore the .csv output
        with open('/dev/null', 'w') as fd:    
            csv_df = self.to_csv(src_info, fd)
        print(csv_df.to_markdown(), file=dest)

    #------------------------------------
    # _numerify_df_indexes
    #-------------------
    
    @classmethod
    def _numerify_df_indexes(cls, df):
        '''
        Go through the given df's columns and index.
        Turn any strings into numbers, if possible,
        else retain the orginals. Returns the df 
        with columns and index changed.
        
        Example:
        
                     '0'      '1'   'Fact'
              '0'   'foo'     10      0
              '1'   'bar'     20      1
        
        will return:
                     0        1     'Fact'
              0    'foo'     10       0
              1    'bar'     20       1
        
        Useful after JSON operations turned numbers
        into strings.
        
        :param df: dataframe to fix
        :type df: pd.DataFrame
        :return the modified df
        :rtype pd.DataFrame
        '''
        
        # Numbers like 0 and 1 are converted to string by json,
        # turn them into nums again
        numeric_cols = cls._numerify_list(list(df.columns))
        df.columns = numeric_cols

        # Same with row labels:
        numeric_idx  = cls._numerify_list(list(df.index))
        df.index = numeric_idx
        
        return df

    #------------------------------------
    # _numerify_list
    #-------------------
    
    @classmethod
    def _numerify_list(cls, the_list):
        '''
        Given a list of strings and or numbers, return
        a new list in which any string that contains a
        number is turned into a numeric value.
        
        Example:
           ['foo', '1', 0] ==> ['foo' 1, 0]
        
        :param the_list: the list to fix
        :type the_list: list[union[int,float,str]]
        :return modified list
        :rtype list[union[int,float,str]]
        '''
        
        new_list = []
        for el in the_list:
            try:
                # Is col nm an int?
                new_list.append(int(el))
            except ValueError:
                # Is it a float?
                try:
                    new_list.append(float(el))
                except ValueError:
                    # Col name is really a string:
                    new_list.append(el)
        return new_list

    
    #------------------------------------
    # __new__
    #-------------------
    
    @staticmethod
    def __new__(cls,
                y_test,
                y_pred,
                **kwargs):
        '''
        Called for two purposes:
        1. To create a ClassifictionResult instance
           under normal circumstances: with y_test
           and y_pred being sequences of class labels,
           truth, and predicted.
           
           For this case the __init__() method will be
           called, and all computations there will occur.
           
           Any kwargs other than the ones in the __init__()
           method's signature will be ignored. 
           
        2. To create an instance with all the computations
           already done. The purpose is just to have an
           object with all information stored as if in the
           final state of the __init__() method. That method
           will not be called. In this case the following
           kwargs must be provided:
           
               - All of ClassifictionResult.ALL_ATTRS
               - An entry for each class label.
           
        Used, for instance, to create ClassificationResult 
        instances that are aggregates of several ClassificationResult
        instances, such as the mean() of a list of results.
        
        :param y_test: true class labels. If None, then __init__
            method is not called, and procedure 2. is assumed,
            and the value is ignored.
        :type y_test: union[None, list[int, str]]
        :param y_pred: predicted class labels. If None, then __init__
            method is not called, and procedure 2. is assumed,
            and the value is ignored.
        :type y_pred: union[None, list[int, str]]
        :param cls_ir_results: dict {cls_nm : ir-res-Series}
        :type cls_ir_results: dict[str : pd.Series
        :param acc: accuracy
        :type acc: float
        :param balanced_acc: balanced accuracy
        :type balanced_acc: float
        :param classes_: list of classes as used in classifier
        :type classes_: optional[list[str]]
        :param cls_nms: the names of classes, which are also this
            object's attribute names for the class-specific IR results.
        :type cls_nms: list[str]
        :param class_label_map: optional map from classifier class
            labels to human readable labels
        :type class_label_map: optional[dict[str : str]]
        :param conf_mat: a classification confusion matrix 
        :type conf_mat: pd.DataFrame 
        :returns a new, initialized instance of ClassificationResult
        :rtype ClassificationResult 
        '''
        # Create empty instance:
        obj = super().__new__(cls)
        
        if y_test is not None and y_pred is not None:
            # Procedure 1:
            obj.__init__(y_test, y_pred, **kwargs)
            return obj

        # Check that all necessary kwargs 
        # are provided:
        try:
            for required_attr in cls.RESULT_ATTRS:
                setattr(obj, required_attr, kwargs[required_attr])
        except KeyError as e:
            raise ValueError(f"Missing argument for raw __new__(): {e}")        

        try:
            for cls_nm in obj.cls_nms:
                setattr(obj, cls_nm, kwargs[cls_nm])
        except KeyError as e:
            raise ValueError(f"Missing argument for raw __new__(): {e}")        
        
        return obj
    
    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        repr_str = f"<ClassificationReport {len(self.classes_)} classes at {hex(id(self))}>"
        return repr_str
    
