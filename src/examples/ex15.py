#ex15.py
#-------
from composes.utils import io_utils
from composes.composition.full_additive import FullAdditive

#training data
train_data = [("good", "car", "good_car"),
              ("good", "book", "good_book")
              ]

#load an argument space
arg_space = io_utils.load("./data/out/ex10.pkl")

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
composed_space  = my_comp.compose([("good", "bike", "good_bike")], 
                                  arg_space)
print "\nComposed space:" 
print composed_space.id2row
print composed_space.cooccurrence_matrix
