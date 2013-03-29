#ex20.py
#-------
from composes.utils import io_utils
from composes.utils import scoring_utils
from composes.similarity.cos import CosSimilarity

#read in a space
my_space = io_utils.load("data/out/ex01.pkl")

#compute similarities of a list of word pairs
fname = "data/in/word_sims.txt"
word_pairs = io_utils.read_tuple_list(fname, fields=[0,1])
predicted = my_space.get_sims(word_pairs, CosSimilarity())

#compute correlations
gold = io_utils.read_list(fname, field=2)
print "Spearman"
print scoring_utils.score(gold, predicted, "spearman")
print "Pearson"
print scoring_utils.score(gold, predicted, "pearson")