#ex01.py
#-------
from composes.semantic_space.space import Space

#create a space from co-occurrence counts in sparse format
my_space = Space.build(data = "./data/in/ex01.sm",
                       rows = "./data/in/ex01.rows",
                       cols = "./data/in/ex01.cols",
                       format = "sm")

#export the space in sparse format
my_space.export("./data/out/ex01", format = "sm")
    
#export the space in dense format
my_space.export("./data/out/ex01", format = "dm")
