#ex15.py
#-------
from composes.utils import io_utils
from composes.composition.full_additive import FullAdditive

#training data
train_data = [("car", "book", "my_car_book"),
              ("book", "book", "2x_book")
              ]

#load an argument space
arg_space = io_utils.load("./data/out/ex01.pkl")

#load a phrase space
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")
print "Training phrase space"
print phrase_space.id2row
print phrase_space.cooccurrence_matrix

#train a FullAdditive model on the data
my_comp = FullAdditive()
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "\nA:", my_comp._mat_a_t.transpose()
print "B:", my_comp._mat_b_t.transpose()

#use the model to compose the train data
composed_space  = my_comp.compose([("car", "book", "composed_car_book"),
                                   ("book", "book", "composed_book_book")], 
                                  arg_space)
print "\nComposed space:" 
print composed_space.id2row
print composed_space.cooccurrence_matrix
