'''
Created on Nov 5, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.dim_reduction.nmf import Nmf  
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction
from composes.composition.full_additive import FullAdditive
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.multiplicative import Multiplicative
from composes.composition.dilation import Dilation 
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.matrix.dense_matrix import DenseMatrix
from composes.transformation.scaling.row_normalization import RowNormalization

import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils


class IntegrationTest(unittest.TestCase):


    data_path = "/home/georgianadinu/work/localtoolkit/toolkit/resource/tutorial/"

    def setUp(self):
        #load a core space
        print "Loading the data..."
        data_path = self.data_path 
        
        space_file = data_path + "CORE_SS.verbnoun.core.pkl"
        self.space = io_utils.load(space_file)

        train_data_file = self.data_path + "ML08_SV_train.txt"
        self.train_data = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])

        test_phrases_file = self.data_path + "ML08nvs_test.txt" 
        self.test_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
        
        print "Reading similarity test data..."
        test_similarity_file = self.data_path + "ML08data_new.txt"
        self.test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
        self.gold = io_utils.read_list(test_similarity_file, field=2)        
        
    def tearDown(self):
        self.space = None
        self.per_space = None
            
    def apply_trans(self, trans):
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        
        print "Applying SVD..."
        self.space = self.space.apply(trans)
    
    def lex_ful_exercise(self, learner_=None):
        
        if learner_ is None:
            learner_ = RidgeRegressionLearner(param=1)
            
        print "Training Lexical Function composition model..."
        comp_model = LexicalFunction(learner = learner_)
        comp_model.train(self.train_data, self.space, self.per_space)

        print "Composing phrases..."
        composed_space = comp_model.compose(self.test_phrases, self.space)
        
        print "Computing similarity with lexical function..."
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())

        #use this composed space to assign similarities
        print "Scoring lexical function..."
        lex_func_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                        scoring_utils.score(self.gold, pred, "pearson"))
        return lex_func_res        
            
    def exercise(self):
        
        lex_func_res = self.lex_ful_exercise()
        print lex_func_res
        
        print "Training Full Additive composition model..."
        comp_model = FullAdditive()
        comp_model.train(self.train_data, self.space, self.per_space)

        composed_space = comp_model.compose(self.test_phrases, self.space)
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())
        
        fadd_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                         scoring_utils.score(self.gold, pred, "pearson"))
        print fadd_res
                
        print "Multiplicative composition model..."
        comp_model = Multiplicative()
        composed_space = comp_model.compose(self.test_phrases, self.space)
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())
        
        mult_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                         scoring_utils.score(self.gold, pred, "pearson"))
        print mult_res
        
        print "Simple additive composition model..."
        comp_model = WeightedAdditive(1, 1)
        composed_space = comp_model.compose(self.test_phrases, self.space.apply(RowNormalization()))
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())
        
        add_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                   scoring_utils.score(self.gold, pred, "pearson"))
        print add_res
                
        print "Simple dilation composition model..."
        comp_model = Dilation()
        composed_space = comp_model.compose(self.test_phrases, self.space)
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())

        dil_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                   scoring_utils.score(self.gold, pred, "pearson"))
        print dil_res
        
        print "Training Weighted Additive composition model..."
        comp_model = WeightedAdditive()
        comp_model.train(self.train_data,  self.space.apply(RowNormalization()),  self.per_space.apply(RowNormalization()))
        composed_space = comp_model.compose(self.test_phrases,  self.space.apply(RowNormalization()))
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())
        
        print "alpha", comp_model.alpha
        print "beta", comp_model.beta
        
        t_add_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                     scoring_utils.score(self.gold, pred, "pearson"))
        print t_add_res
                
        print "Training Dilation composition model..."
        comp_model = Dilation()
        comp_model.train(self.train_data, self.space,  self.per_space)
        composed_space = comp_model.compose(self.test_phrases, self.space)
        pred = composed_space.get_sims(self.test_pairs, CosSimilarity())
        
        print "lambda", comp_model._lambda
        t_dil_res = (scoring_utils.score(self.gold, pred, "spearman"), 
                     scoring_utils.score(self.gold, pred, "pearson"))
        print t_dil_res      
                
        return lex_func_res, fadd_res, mult_res, add_res, dil_res, t_add_res, t_dil_res
            

    
    def ttest_exercise_svd200(self):
        
        self.apply_trans(Svd(200))
    
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        res = self.exercise()
        self.assertAlmostEqual(res[0][0], 0.2386, 3)
        self.assertAlmostEqual(res[2][0], 0.0350, 3)
        self.assertAlmostEqual(res[3][0], 0.1081, 3)
        self.assertAlmostEqual(res[4][0], -0.0505, 3)
        self.assertAlmostEqual(res[5][0], 0.1038, 3)
        self.assertAlmostEqual(res[6][0], -0.0475, 3)
        
    def test_lex_res_lambdas(self):
        self.apply_trans(Svd(200))
    
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )        
        param_range = np.linspace(0,20,41)
        res = []
        for p in param_range:
            print "\nLambda=%.2f" % p
            res.append(self.lex_ful_exercise(RidgeRegressionLearner(param=p)))
            
        #print "\nCV:"
        #res = self.lex_ful_exercise(RidgeRegressionLearner())    
        #self.assertAlmostEqual(res[0], 0.2420, 3)
        #self.assertAlmostEqual(res[1], 0.2537, 3)
                
        for idx,p in enumerate(list(param_range)):
            print "Lambda=%.2f: %.4f %.4f " % (p, res[idx][0], res[idx][1])

    def ttest_exercise_red_full(self):

        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
      
        print "Applying feature selection..."
        self.space = self.space.apply(TopFeatureSelection(2000))
          
        print "Applying SVD..."
        self.space = self.space.apply(Svd(100))
        
        #reading in train data
        train_data_file = self.data_path + "ML08_SV_train.txt"
        train_data = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
        
        print "Training Lexical Function composition model..."
        comp_model = LexicalFunction(learner = RidgeRegressionLearner(param=2))
        comp_model.train(train_data, self.space, self.per_space)

        print "Composing phrases..."
        test_phrases_file = self.data_path + "ML08nvs_test.txt" 
        test_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
        composed_space = comp_model.compose(test_phrases, self.space)
        
        print "Reading similarity test data..."
        test_similarity_file = self.data_path + "ML08data_new.txt"
        test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
        gold = io_utils.read_list(test_similarity_file, field=2)
        
        print "Computing similarity with lexical function..."
        pred = composed_space.get_sims(test_pairs, CosSimilarity())

        #use this composed space to assign similarities
        print "Scoring lexical function...Spearman, Pearson:"
        sp = scoring_utils.score(gold, pred, "spearman")
        prs = scoring_utils.score(gold, pred, "pearson") 
        print sp, prs     
        
        #reduced to FULL
        print "Element shape of the function space:", comp_model.function_space._element_shape
        self.assertAlmostEqual(sp, 0.2961, 3)
        self.assertAlmostEqual(prs, 0.2942, 3)
          
                 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()