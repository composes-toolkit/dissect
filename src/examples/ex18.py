#ex18.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction

#training data 
#trying to learn a "book" function
train_data = [("good_function", "car", "good_car"),
              ("good_function", "book", "good_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex10.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

#train a lexical function model on the data
my_comp = LexicalFunction()
my_comp.train(train_data, arg_space, phrase_space)

#apply the trained model
comp_sp1 = my_comp.compose([("good_function", "car", 
                             "good_car")], 
                           arg_space)

#apply the trained model a second time
comp_sp2 = my_comp.compose([("good_function", "good_car", 
                             "good_good_car")], 
                           comp_sp1)


#print the composed spaces:
print "\nComposed space 1:" 
print comp_sp1.id2row
print comp_sp1.cooccurrence_matrix

print "\nComposed space 2:"
print comp_sp2.id2row
print comp_sp2.cooccurrence_matrix
