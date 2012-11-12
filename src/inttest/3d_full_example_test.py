'''
Created on Nov 5, 2012

@author: georgianadinu
'''
import unittest
from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.dim_reduction.nmf import Nmf 
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction
from composes.composition.full_additive import FullAdditive
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.multiplicative import Multiplicative
from composes.composition.dilation import Dilation 
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.transformation.scaling.row_normalization import RowNormalization

import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils
import composes.utils.log_utils as log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")

class IntegrationTest3D(unittest.TestCase):


    data_path = "/home/georgianadinu/work/localtoolkit/toolkit/resource/tutorial/"
    #data_path = "/home/georgiana.dinu/tutorial/"

    def setUp(self):
        
        log_file = self.data_path + "tmp_log.txt"
        log_utils.config_logging(log_file)
            
        #load a core space
        print "Loading the data..."
        data_path = self.data_path 
        
        space_file = data_path + "CORE_SS.verbnoun.core.pkl"
        self.space = io_utils.load(space_file)

        #space_file = data_path + "GS11data.train_n_vectors_freq10_svd_100_pmi"
        #self.space = Space.build(data = space_file,
        #                            format = "dm"                                
        #                            )
        #space_file = data_path + "GS11data.train_nvn_vectors_freq10_svd_100_pmi"
        #self.per_space = Space.build(data = space_file,
        #                            format = "dm"                                
        #                            )
        
    def tearDown(self):
        self.space = None
        self.per_space = None
            
    def apply_trans(self, dim):
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        
        #print "Applying feature selection..."
        #self.space = self.space.apply(TopFeatureSelection(2000))
        
        print "Applying SVD..."
        self.space = self.space.apply(Svd(dim))
        
    def exercise(self, learner_):
        
        # train 1: VO N -> SVO
        train_data_file = self.data_path + "GS11_SVO_train.txt"
        train_data_vo = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
                
        print "Training Lexical Function composition model STEP1..."
        vo_model = LexicalFunction(learner = learner_)
        vo_model.train(train_data_vo, self.space, self.per_space)
        print "Trained %d distinct VO phrases!" % len(vo_model.function_space.id2row)
        
        # train 2: V N -> VO
        train_data_file = self.data_path + "GS11_V_train.txt"
        train_data_v = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
        
        print "Training Lexical Function composition model STEP2..."
        v_model = LexicalFunction(learner = learner_)
        v_model.train(train_data_v, self.space, vo_model.function_space)

        print "Composing VO phrases..."
        test_phrases_file = self.data_path + "GS11compose1.txt" 
        test_VO_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
        vo_composed_space = v_model.compose(test_VO_phrases, self.space)

        print "Composing SVO sentences..."
        test_phrases_file = self.data_path + "GS11compose2.txt" 
        test_SVO_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])

        expanded_vo_model = LexicalFunction(function_space=vo_composed_space,
                                            intercept=v_model._has_intercept)
        
        svo_composed_space = expanded_vo_model.compose(test_SVO_phrases, self.space)

        print "Reading similarity test data..."
        test_similarity_file = self.data_path + "GS11data_new.txt"
        test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
        gold = io_utils.read_list(test_similarity_file, field=2)
        
        print "Computing similarity with lexical function..."
        pred = svo_composed_space.get_sims(test_pairs, CosSimilarity())

        #use this composed space to assign similarities
        print "\nScoring lexical function..."
        print "Learner:", type(learner_).__name__
        print "Dim CORE space:", self.space.cooccurrence_matrix.shape[1]
        print "Dim PER space:", self.per_space.cooccurrence_matrix.shape[1]
        print scoring_utils.score(gold, pred, "spearman")
        print scoring_utils.score(gold, pred, "pearson")

    def test_exercise_sparse_sparse_svd50(self):
        
        self.apply_trans(100)
        self.space = self.space.apply(RowNormalization())  
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        #self.exercise(LstsqRegressionLearner())
        self.exercise(RidgeRegressionLearner())
        #self.exercise(RidgeRegressionLearner())
 
 
    def tttest_exercise_sparse_sparse_svd50(self):
 
        print "The right one!"
        self.space = self.space.apply(RowNormalization())
        self.per_space = self.per_space.apply(RowNormalization())
               
        self.exercise(RidgeRegressionLearner(param=2))
        
        
    def tttest_exercise_sparse_sparse_svd20_full_per(self):

        self.apply_trans(100)
        self.space = self.space.apply(RowNormalization())    
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        
             
        #self.exercise(LstsqRegressionLearner())
        self.exercise(RidgeRegressionLearner(param=2))
        #self.exercise(RidgeRegressionLearner())
         
    def ttest_exercise_sparse_sparse_svd100(self):   
        
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        print "Applying SVD..."
        self.space = self.space.apply(Svd(100))
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        #self.exercise(LstsqRegressionLearner())
        self.exercise(RidgeRegressionLearner(param=1))
        #self.exercise(RidgeRegressionLearner())

    def ttest_create_new_train_data(self):
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        
        train_data_file = self.data_path + "GS11_SVO_train.txt"
        train_data_vo = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
        list1, list2, list3 = self.valid_train_data_to_lists(train_data_vo,
                                                             (self.space.row2id,
                                                              self.per_space.row2id), 
                                                             self.space.cooccurrence_matrix, 
                                                             self.per_space.cooccurrence_matrix) 
        
        print len(list1)
        f = open(train_data_file + ".pruned", "w")
        for i in xrange(len(list1)):
            print >>f, list1[i], list2[i], list3[i]
        f.close()    
        
        train_data_file = self.data_path + "GS11_V_train.txt"
        train_data_v = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
        list1, list2, list3 = self.valid_train_data_to_lists(train_data_v,
                                                             (self.space.row2id,
                                                              None), 
                                                             self.space.cooccurrence_matrix, 
                                                             self.per_space.cooccurrence_matrix) 
        
        f = open(train_data_file + ".pruned", "w")
        print len(list1)
        for i in xrange(len(list1)):
            print >>f, list1[i], list2[i], list3[i]
        f.close()      
        
    def valid_train_data_to_lists(self, data, (row2id2, row2id3), arg_mat, phrase_mat):

        list1 = []
        list2 = []
        list3 = []
         
        j = 0
        for i in xrange(len(data)):
            sample = data[i]
            
            cond = True
            
            if not row2id2 is None:
                cond = cond and sample[1] in row2id2
            
            if not row2id3 is None:
                cond = cond and sample[2] in row2id3
                cond = cond and phrase_mat[row2id3[sample[2]]].mat.nnz >= 10

            if cond:
                list1.append(sample[0]) 
                list2.append(sample[1])
                list3.append(sample[2])
                j += 1
    
        if i + 1 != j:
            print ("%d (out of %d) lines are ignored because one of the elements is not found in its semantic space"
                 % ((i + 1) - j, (i + 1)))
            
        if not list1:
            raise ValueError("No valid training data found!")
        
        return list1, list2, list3
    
    def ttest_exercise_dense_dense_svd100(self):   
        
        self.space.to_dense()
        
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        print "Applying SVD..."
        self.space = self.space.apply(Svd(100))
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        
        self.exercise(LstsqRegressionLearner())
        #self.exercise(RidgeRegressionLearner(param=2))
        #self.exercise(RidgeRegressionLearner())
        
    def ttest_exercise_sparse_sparse_svd150(self):   
        
        self.apply_trans(150)
    
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        
        self.exercise(LstsqRegressionLearner())
        self.exercise(RidgeRegressionLearner(param=2))
        self.exercise(RidgeRegressionLearner())
             
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()   