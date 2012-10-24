#ex02.py
#-------
from composes.semantic_space.space import Space
from composes.utils import io_utils

#create a space from co-occurrence counts in sparse format
my_space = Space.build(data = "./data/in/ex01.sm",
                       rows = "./data/in/ex01.rows",
                       cols = "./data/in/ex01.cols",
                       format = "sm")

#print the co-occurrence matrix of the space
print my_space.cooccurrence_matrix

#save the Space object in pickle format
io_utils.save(my_space, "./data/out/ex01.pkl")
    
#load the saved object
my_space2 = io_utils.load("./data/out/ex01.pkl")

#print the co-occurrence matrix of the loaded space
print my_space2.cooccurrence_matrix

