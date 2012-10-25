from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd 
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection 
from composes.composition.lexical_function import LexicalFunction 
import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils

#load a core space
core_space_file = "/mnt/cimec-storage-sata/users/thenghia.pham/data/tutorial/raw_core/CORE_SS.verbnoun.core.pkl"
space = io_utils.load(core_space_file)

#apply ppmi weighting
space = space.apply(PpmiWeighting())

#do feature selection
space = space.apply(TopFeatureSelection(2000))

#apply SVD
space = space.apply(Svd(100))

#create peripheral space
per_space_path = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/per_in/"
per_space = PeripheralSpace.build(data = per_space_path + "per.raw.SV.sm",
                                  rows = per_space_path + "per.raw.SV.rows",
                                  cols = per_space_path + "per.raw.SV.cols"                                
                                  )

#train a composition model
train_data_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/per_in/ML08_SV_train.txt"
train_data = io_utils.read_list(train_data_file)

comp_model = LexicalFunction()
comp_model.train(train_data, space, per_space)


#use it to compose the phrases we need
test_phrases_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/test/ML10/ML10_VO_test.txt" 
test_phrases = io_utils.read_list(test_phrases_file)
composed_space = comp_model.compose(test_phrases, space, per_space)


#use this composed space to assign similarities
test_similarity_file = "/mnt/cimec-storage-sata/users/georgiana.dinu/COLING/test/ML10/ML10data_new.txt" 
pred = []
gold = []
with open(test_similarity_file) as instream:
    for line in instream: 
        [w1, w2, g] = line.split()
        gold.append(g)
        pred.append(composed_space.get_sim(w1, w2, CosSimilarity()))

print scoring_utils.score(gold, pred, "spearman")
                    
                    



