from composes.utils import io_utils
from composes.composition.weighted_additive import WeightedAdditive

# instantiate a weighted additive model
my_comp = WeightedAdditive(alpha = 1, beta = 1)

#save it to binary
io_utils.save(my_comp, "./data/out/model01.pkl")

#print its parameters
my_comp.export("./data/out/model01.params")

