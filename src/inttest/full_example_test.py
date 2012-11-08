'''
Created on Nov 5, 2012

@author: georgianadinu
'''
import unittest
from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd 
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction
from composes.composition.full_additive import FullAdditive
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.multiplicative import Multiplicative
from composes.composition.dilation import Dilation 
from composes.utils.regression_learner import RidgeRegressionLearner

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

    def tearDown(self):
        self.space = None
        self.per_space = None
            
    def apply_trans(self):
        print "Applying PPMI..."
        self.space = self.space.apply(PpmiWeighting())
        
        print "Applying feature selection..."
        self.space = self.space.apply(TopFeatureSelection(2000))
        
        print "Applying SVD..."
        self.space = self.space.apply(Svd(100))
        
    def exercise(self):
        
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
        print "Scoring lexical function..."
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.22605, 3)
        
        print "Training Full Additive composition model..."
        comp_model = FullAdditive(learner = RidgeRegressionLearner(param=2))
        comp_model.train(train_data, self.space, self.per_space)
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), -0.0924, 3)
        #self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), -0.0038, 3)
        
        print "Multiplicative composition model..."
        comp_model = Multiplicative()
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.02501, 3)

        print "Simple additive composition model..."
        comp_model = WeightedAdditive(1, 1)
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.03076, 3)
        
        print "Simple dilation composition model..."
        comp_model = Dilation()
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.01670, 3)

        print "Training Weighted Additive composition model..."
        comp_model = WeightedAdditive()
        comp_model.train(train_data, self.space, self.per_space)
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        
        self.assertAlmostEqual(comp_model.alpha, 0.01334, 3)
        self.assertAlmostEqual(comp_model.beta, 0.02019, 3)
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.01814, 3)
        
        print "Training Dilation composition model..."
        comp_model = Dilation()
        comp_model.train(train_data, self.space, self.per_space)
        composed_space = comp_model.compose(test_phrases, self.space)
        pred = composed_space.get_sims(test_pairs, CosSimilarity())
        
        self.assertAlmostEqual(comp_model._lambda, 0.05064, 3)
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.02357, 3)
                
    def test_exercise_sparse_sparse(self):
        
        self.apply_trans()
    
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        self.exercise()

    def test_exercise_dense_sparse(self):

        self.space.to_dense()
        self.apply_trans()
        
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        
        self.exercise()
        
    def test_exercise_dense_dense(self):
        
        self.space.to_dense()
        self.apply_trans()
        print "Creating peripheral space.."
        
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        self.per_space.to_dense()
        
        self.exercise()

    def test_exercise_sparse_dense(self):
        
        self.apply_trans()
            
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        self.per_space.to_dense()
        
        self.exercise()        

    def test_exercise_red_full(self):

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
        print "Scoring lexical function..."
        print scoring_utils.score(gold, pred, "spearman")     
        print scoring_utils.score(gold, pred, "pearson")
        
        #reduced to FULL
        print "Element shape of the function space:", comp_model.function_space._element_shape
        self.assertAlmostEqual(scoring_utils.score(gold, pred, "spearman"), 0.2933, 3)

          
    def ttest_exercise_full_red(self):
   
        full_space = self.space
        
        self.apply_trans()
            
        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        self.per_space.to_sparse()
        #reading in train data
        train_data_file = self.data_path + "ML08_SV_train.txt"
        train_data = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])
        
        print "Training Lexical Function composition model..."
        comp_model = LexicalFunction(learner = RidgeRegressionLearner(param=2))
        comp_model.train(train_data, full_space, self.per_space)

        print "Composing phrases..."
        test_phrases_file = self.data_path + "ML08nvs_test.txt" 
        test_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
        composed_space = comp_model.compose(test_phrases, full_space)
        
        print "Reading similarity test data..."
        test_similarity_file = self.data_path + "ML08data_new.txt"
        test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
        gold = io_utils.read_list(test_similarity_file, field=2)
        
        print "Computing similarity with lexical function..."
        pred = composed_space.get_sims(test_pairs, CosSimilarity())

        #use this composed space to assign similarities
        print "Scoring lexical function..."
        print scoring_utils.score(gold, pred, "spearman")
        print scoring_utils.score(gold, pred, "pearson")
        
    def ttest_exercise_full_full(self):

        print "Creating peripheral space.."
        self.per_space = PeripheralSpace.build(self.space,
                                          data = self.data_path + "per.raw.SV.sm",
                                          cols = self.data_path + "per.raw.SV.cols",
                                          format = "sm"                                
                                          )
        
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
        print "Scoring lexical function..."
        print scoring_utils.score(gold, pred, "spearman")     
        print scoring_utils.score(gold, pred, "pearson")
                 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()