from composes.utils import io_utils
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting


#load a space and apply Ppmi on it
my_space = io_utils.load("./data/out/ex01.pkl")
my_space = my_space.apply(PpmiWeighting())

print my_space.cooccurrence_matrix
print my_space.id2row

#create a peripheral space 
my_per_space = PeripheralSpace.build(my_space,
                                     data="./data/in/ex05.sm",
                                     cols="./data/in/ex05.cols",
                                     format="sm")

print my_per_space.cooccurrence_matrix
print my_per_space.id2row

#save the space
io_utils.save(my_per_space, "./data/out/PER_SS.ex05.pkl")

