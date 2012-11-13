'''
Created on Nov 5, 2012

@author: georgianadinu
'''
import unittest
from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import RidgeRegressionLearner
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
        
    def tearDown(self):
        self.space = None
        self.per_space = None
            
    def apply_trans(self, dim):
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        
        print "Applying SVD..."
        self.space = self.space.apply(Svd(dim))
        
        print "Row-normalizing..."
        self.space = self.space.apply(RowNormalization())
        
    def exercise(self, learner_):
        
        # train 1: VO N -> SVO
        train_data_file = self.data_path + "GS11_SVO_train.txt"
        train_data_vo = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
                
        print "Training Lexical Function composition model STEP1..."
        vo_model = LexicalFunction(learner = learner_)
        vo_model._MIN_SAMPLES = 5
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
        sp = scoring_utils.score(gold, pred, "spearman")
        prs =  scoring_utils.score(gold, pred, "pearson")
        return sp, prs 

    def test_exercise_dense_dense_svd100(self):
        
        self.apply_trans(100)
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
 
        res = self.exercise(RidgeRegressionLearner(param=2))
        print "SVD 100, row-norm, RidgeRegression-lambda=2, MIN_SAMPLES=5:", res 
        self.assertAlmostEqual(res[0], 0.3404, 3)
        self.assertAlmostEqual(res[1], 0.3122, 3)
        
        #print self.exercise(RidgeRegressionLearner())
        #(0.32258989594699261, 0.29441216345028631)
 
    def test_exercise_dense_dense_svd10(self):
        
        self.apply_trans(10)
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
 
        res = self.exercise(RidgeRegressionLearner())
        self.assertAlmostEqual(res[0], 0.2977, 3)
        self.assertAlmostEqual(res[1], 0.2175, 3)
        
    def test_exercise_sparse_sparse_svd100(self):
        
        self.apply_trans(10)
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        self.space.to_sparse()
        self.per_space.to_sparse()
        
        res = self.exercise(RidgeRegressionLearner(param=2))
        print "SVD 100, row-norm, RidgeRegression-lambda=2, MIN_SAMPLES=5:", res 
        self.assertAlmostEqual(res[0], 0.3053, 3)
        
 
    def test_exercise_sparse_dense_svd100(self):
        
        self.apply_trans(10)
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        self.space.to_sparse()
        
        res = self.exercise(RidgeRegressionLearner(param=2))
        print "SVD 100, row-norm, RidgeRegression-lambda=2, MIN_SAMPLES=5:", res 
        self.assertAlmostEqual(res[0], 0.3053, 3)
         
    def test_exercise_dense_sparse_svd100(self):
        
        self.apply_trans(10)
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SVO.sm",
                                          cols = self.data_path + "per.raw.SVO.cols",
                                          format = "sm"                                
                                          )
        self.per_space.to_sparse()
        
        res = self.exercise(RidgeRegressionLearner(param=2))
        print "SVD 100, row-norm, RidgeRegression-lambda=2, MIN_SAMPLES=5:", res 
        self.assertAlmostEqual(res[0], 0.3053, 3)
        
        
             
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()   