#ex17.py
#-------
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import RidgeRegressionLearner

#training data 
#trying to learn a "book" function
train_data = [("book", "car", "my_car_book"),
              ("book", "book", "my_special_book")
              ]

#load argument and phrase space
arg_space = io_utils.load("./data/out/ex01.pkl")
phrase_space = io_utils.load("data/out/PHRASE_SS.ex10.pkl")

print "\nDefault LSTSQ regression"
my_comp = LexicalFunction()
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "Lexical function space:" 
print my_comp.function_space.id2row
cooc_mat = my_comp.function_space.cooccurrence_matrix
cooc_mat.reshape(my_comp.function_space.element_shape)
print cooc_mat

print "\nRidge Regression with lambda = 1"
rr_learner=RidgeRegressionLearner(param = 1,
                                  intercept = False, 
                                  crossvalidation=False)
my_comp = LexicalFunction(learner = rr_learner)
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "Lexical function space:" 
print my_comp.function_space.id2row
cooc_mat = my_comp.function_space.cooccurrence_matrix
cooc_mat.reshape(my_comp.function_space.element_shape)
print cooc_mat
