import logging
from composes.composition.lexical_function import LexicalFunction
from composes.matrix.linalg import Linalg
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.similarity.cos import CosSimilarity
from composes.similarity.lin import LinSimilarity
from composes.transformation.dim_reduction.nmf import Nmf
from composes.semantic_space.space import Space
from thesisgenerator.plugins.thesaurus_loader import Thesaurus


def julie_transform(input):
    '''african/J:amod-HEAD:ancestry -> african_ancestry'''
    noun = input.split(':')[-1]
    adj = input.split('/')[0]
    return '{}_{}'.format(adj, noun)


def translate_byblo_to_dissect(events_file, row_transform=lambda x: x):
    """
    Translated Byblo-made vectors file to dissect format in the absence of features/entries files
    :param events_file: path to byblo-made vectors
    :type events_file: str
    :return: prefix of dissect-compatible data files
    :rtype: str
    """
    # remove duplicate head noun vectors, converting to a dissect sparse matrix format
    t = Thesaurus([events_file], aggressive_lowercasing=False)
    t.to_dissect_sparse_files('{}.uniq'.format(ANs_events_file), row_transform=row_transform)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")
logging.info('Starting')
noun_events_file = './data/in/julie.nouns.vectors'
ANs_events_file = './data/in/julie.ANs.vectors'
an_events_file = '{}.uniq'.format(ANs_events_file)
noun_events_file = '{}.uniq'.format(noun_events_file)
#noun_events_file = translate_byblo_to_dissect(noun_events_file)
#an_events_file = translate_byblo_to_dissect(ANs_events_file, row_transform=julie_transform)

#my_space = Thesaurus([noun_events_file], aggressive_lowercasing=False).to_dissect_core_space()
my_space = Space.build(data="{}.sm".format(noun_events_file),
                       rows="{}.rows".format(noun_events_file),
                       cols="{}.cols".format(noun_events_file),
                       format="sm")

Linalg._NMF_MAX_ITER = 2
my_space = my_space.apply(Nmf(10))


#create a peripheral space
my_per_space = PeripheralSpace.build(my_space,
                                     data="{}.sm".format(an_events_file),
                                     rows="{}.rows".format(an_events_file),
                                     # The columns of the peripheral space have to be identical to those
                                     # in the core space (including their order)!
                                     cols="{}.cols".format(noun_events_file),
                                     format="sm")

lin = LinSimilarity()

#for a, b in product(my_space.row2id, my_space.row2id):
#    sim = my_space.get_sim(a, b, lin)
#    if sim > 0:
#        logging.info('%s <> %s = %0.2f', a, b, sim)
#
#for i, noun in enumerate(my_space.row2id):
#    logging.info(my_space.get_neighbours(noun, 2, lin))
#    #logging.info(my_space.get_neighbours(noun, 2, lin, space2=my_per_space))



#export the space in sparse format
my_space.export("./data/out/ex01", format="dm")

#my_space = my_space.apply(PpmiWeighting())
#my_space = my_space.apply(Svd(5))



# use the model to compose words in my_space
all_data = []
for an in my_per_space._row2id:
    adj, noun = an.split('_')
    print adj
    all_data.append((adj, noun, '%s_%s' % (adj, noun)))

#train a lexical function model on the data
my_comp = LexicalFunction()
my_comp.train(all_data[:80], my_space, my_per_space)

#similarity within the learned functional space
composed_space = my_comp.compose(all_data[80:], my_space)
#return (composed_space.row2id, composed_space.cooccurrence_matrix)
logging.info(my_comp.function_space.get_sim("african_army", "african_army", CosSimilarity()))

# this computes the sim by a brute force
composed_space.get_neighbours('african_army', 4, CosSimilarity(), space2=my_per_space)
composed_space.get_row('african_army').mat
from composes.utils.io_utils import save

save(composed_space, 'data/out/trained_AN.pkl')
logging.info('Done')

