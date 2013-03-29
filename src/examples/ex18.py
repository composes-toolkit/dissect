#ex18.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction

#training data 
#trying to learn a "book" function
train_data = [("book_function", "car", "my_car_book"),
              ("book_function", "book", "2x_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex01.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

#train a lexical function model on the data
my_comp = LexicalFunction()
my_comp._MIN_SAMPLES = 1
my_comp.train(train_data, arg_space, phrase_space)

#apply the trained model
comp_sp1 = my_comp.compose([("book_function", "car", 
                             "composed_book_car")], 
                           arg_space)

#apply the trained model a second time
comp_sp2 = my_comp.compose([("book_function", "composed_book_car", 
                             "composed_book_book_car")], 
                           comp_sp1)


#print the composed spaces:
print "\nComposed space 1:" 
print comp_sp1.id2row
print comp_sp1.cooccurrence_matrix

print "\nComposed space 2:"
print comp_sp2.id2row
print comp_sp2.cooccurrence_matrix
