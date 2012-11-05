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

#load a core space
print "Loading the data..."
data_path = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/"

space_file = data_path + "CORE_SS.verbnoun.core.pkl"
space = io_utils.load(space_file)

print "Applying PPMI..."
space = space.apply(PpmiWeighting())

print "Applying feature selection..."
space = space.apply(TopFeatureSelection(2000))

print "Applying SVD..."
space = space.apply(Svd(100))

print "Creating peripheral space.."
per_space = PeripheralSpace.build(space,
                                  data = data_path + "per.raw.SV.sm",
                                  cols = data_path + "per.raw.SV.cols",
                                  format = "sm"                                
                                  )

#reading in train data
train_data_file = data_path + "ML08_SV_train.txt"
train_data = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])

print "Training Lexical Function composition model..."
comp_model = LexicalFunction(learner = RidgeRegressionLearner(param=2))
comp_model.train(train_data, space, per_space)

print "Composing phrases..."
test_phrases_file = data_path + "ML08nvs_test.txt" 
test_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
composed_space = comp_model.compose(test_phrases, space)

print "Reading similarity test data..."
test_similarity_file = data_path + "ML08data_new.txt"
test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
gold = io_utils.read_list(test_similarity_file, field=2)

print "Computing similarity with lexical function..."
pred = composed_space.get_sims(test_pairs, CosSimilarity())

#use this composed space to assign similarities
print "Scoring lexical function..."
print scoring_utils.score(gold, pred, "spearman")
                    

print "Training Full Additive composition model..."
comp_model = FullAdditive(learner = RidgeRegressionLearner(param=2))
comp_model.train(train_data, space, per_space)
composed_space = comp_model.compose(test_phrases, space)
pred = composed_space.get_sims(test_pairs, CosSimilarity())
print scoring_utils.score(gold, pred, "spearman")

print "Training Weighted Additive composition model..."
comp_model = WeightedAdditive()
comp_model.train(train_data, space, per_space)
composed_space = comp_model.compose(test_phrases, space)
pred = composed_space.get_sims(test_pairs, CosSimilarity())
print scoring_utils.score(gold, pred, "spearman")

print "Training Dilation composition model..."
comp_model = Dilation()
comp_model.train(train_data, space, per_space)
composed_space = comp_model.compose(test_phrases, space)
pred = composed_space.get_sims(test_pairs, CosSimilarity())
print scoring_utils.score(gold, pred, "spearman")

print "Multiplicative composition model..."
comp_model = Multiplicative()
composed_space = comp_model.compose(test_phrases, space)
pred = composed_space.get_sims(test_pairs, CosSimilarity())
print scoring_utils.score(gold, pred, "spearman")
