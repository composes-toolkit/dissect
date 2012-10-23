#ex18.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction

#training data 
#trying to learn a "book" function
train_data = [("book", "car", "my_car_book"),
              ("book", "book", "my_special_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex01.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

#train an lexical function model on the data
my_comp = LexicalFunction()
my_comp.train(train_data, arg_space, phrase_space)

#apply the trained model
composed_space1 = my_comp.compose([("book", "car", "composed_book_car")], 
                                  arg_space)

#apply the trained model a second time
composed_space2 = my_comp.compose([("book", "composed_book_car", "composed_book_book_car")], 
                                   composed_space1)


#print the composed spaces:
print "\nComposed space 1:" 
print composed_space1.id2row
print composed_space1.cooccurrence_matrix

print "\nComposed space 2:"
print composed_space2.id2row
print composed_space2.cooccurrence_matrix
