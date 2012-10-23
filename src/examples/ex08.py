from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity

#load a space
my_space = io_utils.load("./data/out/ex01.pkl")

#get the top 2 neighbours of "car"
print my_space.get_neighbours("car", 2, CosSimilarity())
