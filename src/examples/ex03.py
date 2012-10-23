#ex03.py
#-------
from composes.utils import io_utils
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting

#create a space from co-occurrence counts in sparse format
my_space = io_utils.load("./data/out/ex01.pkl")

#print the co-occurrence matrix of the space
print my_space.cooccurrence_matrix

#apply ppmi weighting
my_space = my_space.apply(PpmiWeighting())

#print the co-occurrence matrix of the transformed space
print my_space.cooccurrence_matrix

