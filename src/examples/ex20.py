#ex20.py
#-------
from composes.utils import io_utils
from composes.utils import scoring_utils
from composes.utils import log_utils
from composes.similarity.cos import CosSimilarity

#setting up a logging file
log_utils.config_logging("data/out/log_ex20.txt")

#1. IO UTILS
fname = "data/in/sim_data3.txt"

#reads the first two columns of the input file as a list of tuples
pair_list = io_utils.read_tuple_list(fname, fields=[0,1])
print pair_list

#reads ALL the columns of the input file as a list of tuples
elem_list = io_utils.read_tuple_list(fname)
print elem_list


#reads a specific column of a file as a list
sim_column = io_utils.read_list(fname, field=2)
print sim_column

#reads the lines of file as a list (each line, one element)
data = io_utils.read_list(fname)
print data


#2. SIMILARITY QUERY FOR A LIST OF WORDS PAIRS
my_space = io_utils.load("./data/out/ex01.pkl")

#similarity of two words
print "\nSimilarity of car book"
print my_space.get_sim("car", "book", CosSimilarity())

#similarity for a list of word pairs (list of tuples)
print "Pair list"
print pair_list
print "Similarity of pair list"
print my_space.get_sims(pair_list, CosSimilarity())


#3. SCORING UTILS (CORRELATION MEASURES)
predicted = my_space.get_sims(pair_list, CosSimilarity())
gold = io_utils.read_list(fname, field=4)

print "\nGold"
print gold
print "Predicted"
print predicted
print "Spearman"
print scoring_utils.score(gold, predicted, "spearman")

