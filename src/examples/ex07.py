#ex07.py
#-------
from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity

#load two spaces
my_space = io_utils.load("./data/out/ex01.pkl")
my_per_space = io_utils.load("./data/out/PER_SS.ex05.pkl")

print my_space.id2row
print my_per_space.id2row

##compute similarity between a word and a phrase in the two spaces
print my_space.get_sim("car", "sports_car", CosSimilarity(), 
                       space2 = my_per_space)
