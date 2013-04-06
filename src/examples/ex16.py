#ex16.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction
from composes.similarity.cos import CosSimilarity

#training data 
#trying to learn a "good" function
train_data = [("good_function", "car", "good_car"),
              ("good_function", "book", "good_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex10.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

#train a lexical function model on the data
my_comp = LexicalFunction()
my_comp._MIN_SAMPLES = 1
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "\nLexical function space:" 
print my_comp.function_space.id2row
cooc_mat = my_comp.function_space.cooccurrence_matrix
cooc_mat.reshape(my_comp.function_space.element_shape)
print cooc_mat

#similarity within the learned functional space
print "\nSimilarity between good and good in the function space:" 
print my_comp.function_space.get_sim("good_function", "good_function", 
                                     CosSimilarity())