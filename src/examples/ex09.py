from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity

#load two spaces
my_space = io_utils.load("./data/out/ex01.pkl")
my_per_space = io_utils.load("./data/out/PER_SS.ex05.pkl")

#get the top two neighbours of "car" in my_per_space 
print my_space.get_neighbours("car", 2, CosSimilarity(), neighbour_space = my_per_space)

