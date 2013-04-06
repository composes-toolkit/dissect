#ex12.py
#-------
from composes.utils import io_utils

#load a previously saved weighted additive model
my_comp = io_utils.load("./data/out/model01.pkl")

#print its parameters
print "alpha:", my_comp.alpha
print "beta:", my_comp.beta

#load two spaces
my_space = io_utils.load("./data/out/ex10.pkl")
my_per_space = io_utils.load("./data/out/PER_SS.ex05.pkl")

#apply the composition model to them
composed_space = my_comp.compose([("good", "history_book", "good_history_book")], 
                                 (my_space, my_per_space))

print composed_space.id2row
print composed_space.cooccurrence_matrix



