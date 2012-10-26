from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd 
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction 
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.dilation import Dilation
import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils

#load a core space
print "Loading the data..."
#core_space_file = "/mnt/cimec-storage-sata/users/thenghia.pham/data/tutorial/raw_core/CORE_SS.verbnoun.core.pkl"
#space = io_utils.load(core_space_file)

space_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/core_in/core.raw.pkl"
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
per_space_path = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/per_in/"
per_space = PeripheralSpace.build(space,
                                  data = per_space_path + "per.raw.SV.sm",
                                  cols = per_space_path + "per.raw.SV.cols",
                                  format = "sm"                                
                                  )


#train a composition model
train_data_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/per_in/ML08_SV_train.txt"
train_data = io_utils.read_tuple_list(train_data_file, 3)

print "Training composition model..."
comp_model = LexicalFunction()
comp_model.train(train_data, space, per_space)

print "Composing phrases..."
#use it to compose the phrases we need
test_phrases_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/test/ML08/ML08nvs_test.txt" 
test_phrases = io_utils.read_tuple_list(test_phrases_file, 3)
composed_space = comp_model.compose(test_phrases, space)


print "Computing similarity..."
#use this composed space to assign similarities
test_similarity_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/test/ML08/ML08data_new.txt" 
pred = []
gold = []
with open(test_similarity_file) as instream:
    for line in instream: 
        [w1, w2, g] = line.split()
        gold.append(g)
        pred.append(composed_space.get_sim(w1, w2, CosSimilarity()))

print "Scoring lexical function..."
print scoring_utils.score(gold, pred, "spearman")
                    

