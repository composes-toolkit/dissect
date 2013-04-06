#ex17.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import RidgeRegressionLearner

#training data 
#trying to learn a "good" function
train_data = [("good_function", "car", "good_car"),
              ("good_function", "book", "good_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex10.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

print "\nDefault regression:"
my_comp = LexicalFunction()
print type(my_comp.regression_learner).__name__
my_comp._MIN_SAMPLES = 1
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "Lexical function space:" 
print my_comp.function_space.id2row
cooc_mat = my_comp.function_space.cooccurrence_matrix
cooc_mat.reshape(my_comp.function_space.element_shape)
print cooc_mat

print "\nRidge Regression with lambda = 2"
rr_learner=RidgeRegressionLearner(param = 2,
                                  intercept = False, 
                                  crossvalidation=False)
my_comp = LexicalFunction(learner = rr_learner)
my_comp._MIN_SAMPLES = 1
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "Lexical function space:" 
print my_comp.function_space.id2row
cooc_mat = my_comp.function_space.cooccurrence_matrix
cooc_mat.reshape(my_comp.function_space.element_shape)
print cooc_mat
