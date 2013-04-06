#ex10.py
#-------
from composes.utils import io_utils
from composes.composition.weighted_additive import WeightedAdditive

#load a space
my_space = io_utils.load("./data/out/ex10.pkl")

print my_space.id2row
print my_space.cooccurrence_matrix

# instantiate a weighted additive model
my_comp = WeightedAdditive(alpha = 1, beta = 1)

# use the model to compose words in my_space
composed_space = my_comp.compose([("good", "book", "good_book"),
                                  ("good", "car", "good_car")], 
                                 my_space)

print composed_space.id2row
print composed_space.cooccurrence_matrix

#save the composed space
io_utils.save(composed_space, "data/out/PHRASE_SS.ex10.pkl")


