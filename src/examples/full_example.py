from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd 
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction 
from composes.composition.weighted_additive import WeightedAdditive
import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils

#load a core space
print "Loading the data..."
space_file = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/CORE_SS.verbnoun.core.pkl"
space = io_utils.load(space_file)

print "Applying PPMI..."
#apply ppmi weighting
space = space.apply(PpmiWeighting())

print "Applying feature selection..."
#do feature selection
space = space.apply(TopFeatureSelection(2000))

print "Applying SVD..."
#apply SVD
space = space.apply(Svd(100))

print "Creating peripheral space.."
#create peripheral space
per_space_path = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/"
per_space = PeripheralSpace.build(space,
                                  data = per_space_path + "per.raw.SV.sm",
                                  cols = per_space_path + "per.raw.SV.cols",
                                  format = "sm"                                
                                  )

#train a composition model
train_data_file = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08_SV_train.txt"
train_data = io_utils.read_tuple_list(train_data_file, fields=[0,1,2])

print "Training Lexical Function composition model..."
comp_model = LexicalFunction()
comp_model.train(train_data, space, per_space)

print "Composing phrases..."
#use it to compose the phrases we need
test_phrases_file = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08nvs_test.txt" 
test_phrases = io_utils.read_tuple_list(test_phrases_file, fields=[0,1,2])
composed_space = comp_model.compose(test_phrases, space)


print "Reading similarity test data..."
test_similarity_file = "/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial/ML08data_new.txt"
test_pairs = io_utils.read_tuple_list(test_similarity_file, fields=[0,1])
gold = io_utils.read_list(test_similarity_file, field=2)

print "Computing similarity with lexical function..."
pred = composed_space.get_sims(test_pairs, CosSimilarity())

#use this composed space to assign similarities
print "Scoring lexical function..."
print scoring_utils.score(gold, pred, "spearman")
                    
print "Testing additive model.."
comp_model = WeightedAdditive(1,1)
composed_space = comp_model.compose(test_phrases, space)

print "Computing similarity with weighted_additive(1,1)..."
pred = composed_space.get_sims(test_pairs, CosSimilarity())

print "Scoring additive..."
print scoring_utils.score(gold, pred, "spearman")


