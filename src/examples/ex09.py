#ex09.py
#-------
from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity

#load two spaces
my_space = io_utils.load("./data/out/ex01.pkl")
my_per_space = io_utils.load("./data/out/PER_SS.ex05.pkl")

print my_space.id2row
print my_space.cooccurrence_matrix
print my_per_space.id2row
print my_per_space.cooccurrence_matrix

#get the top two neighbours of "car" in a peripheral space 
print my_space.get_neighbours("car", 2, CosSimilarity(), 
                              space2 = my_per_space)

