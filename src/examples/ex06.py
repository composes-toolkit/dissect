from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity

#load a space
my_space = io_utils.load("./data/out/ex01.pkl")

print my_space.cooccurrence_matrix
print my_space.id2row

#compute similarity between two words in the space 
print my_space.get_sim("car", "book", CosSimilarity())
