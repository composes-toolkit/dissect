#ex03.py
#-------
from composes.utils import io_utils
from composes.transformation.dim_reduction.svd import Svd

#load a space
my_space = io_utils.load("./data/out/ex01.pkl")

#print the co-occurrence matrix and the columns of the space
print my_space.cooccurrence_matrix
print my_space.id2column

#apply svd reduction
my_space = my_space.apply(Svd(2))

#print the transformed space
print my_space.cooccurrence_matrix
print my_space.id2column
