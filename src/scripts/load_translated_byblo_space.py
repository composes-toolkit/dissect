import logging
from composes.composition.lexical_function import LexicalFunction
from composes.matrix.linalg import Linalg
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.similarity.cos import CosSimilarity
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
    uniq_events_file = '{}.uniq'.format(events_file)
    # remove duplicate head noun vectors, converting to a dissect sparse matrix format
    t = Thesaurus([events_file], aggressive_lowercasing=False)
    rows = set(t.keys())
    columns = set(feature for vector in t.values() for (feature, count) in vector)

    with open('{0}.rows'.format(uniq_events_file), 'w+b') as outfile:
        for line in rows:
            outfile.write(row_transform(line) + '\n')

    with open(events_file) as infile, open('{0}.sm'.format(uniq_events_file), 'w+b') as outfile:
        for line in infile:
            tokens = line.split('\t')
            if tokens[0] in rows:
                rows.remove(tokens[0])
                tokens[0] = row_transform(tokens[0])
                outfile.write(' '.join(tokens))
                # write dissect columns file

    with open('{}.cols'.format(uniq_events_file), 'w+b') as outfile:
        for feature in sorted(columns):
            outfile.write('{}\n'.format(feature))

    return uniq_events_file


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

noun_events_file = './data/in/julie.nouns.vectors'
ANs_events_file = './data/in/julie.ANs.vectors'
#noun_events_file = translate_byblo_to_dissect(nouns_events_file)
#an_events_file = translate_byblo_to_dissect(ANs_events_file, row_transform=julie_transform)
an_events_file = '{}.uniq'.format(ANs_events_file)
noun_events_file = '{}.uniq'.format(noun_events_file)

my_space = Space.build(data="{}.sm".format(noun_events_file),
                       rows="{}.rows".format(noun_events_file),
                       cols="{}.cols".format(noun_events_file),
                       format="sm")


#create a peripheral space
my_per_space = PeripheralSpace.build(my_space,
                                     data="{}.sm".format(an_events_file),
                                     rows="{}.rows".format(an_events_file),
                                     # The columns of the peripheral space have to be identical to those
                                     # in the core space (including their order)!
                                     cols="{}.cols".format(noun_events_file),
                                     format="sm")
Linalg._NMF_MAX_ITER = 2
my_space = my_space.apply(Nmf(10))

#export the space in sparse format
#my_space.export("./data/out/ex01", format="sm")

#my_space = my_space.apply(PpmiWeighting())
#my_space = my_space.apply(Svd(5))



# use the model to compose words in my_space
all_data = []
for an in my_per_space._row2id:
    a, n = an.split('_')
    all_data.append((a, n, '%s_%s' % (a, n)))

#train a lexical function model on the data
my_comp = LexicalFunction()
my_comp.train(all_data[:50], my_space, my_per_space)

#print its parameters
#print "\nLexical function space:"
#print my_comp.function_space.id2row
#cooc_mat = my_comp.function_space.cooccurrence_matrix
#cooc_mat.reshape(my_comp.function_space.element_shape)
#print cooc_mat

#similarity within the learned functional space
print "\nSimilarity between good and good in the function space:"
print my_comp.function_space.get_sim("african_army", "african_army", CosSimilarity())

#print composed_space.id2row
#print composed_space.cooccurrence_matrix
#save the composed space
#io_utils.save(composed_space, "data/out/PHRASE_SS.ex10.pkl")
#print my_space.get_neighbours("bachelor/N", 2, CosSimilarity())
#print my_space.get_neighbours("bachelor/N", 2, CosSimilarity(), space2=composed_space)
#print composed_space.get_neighbours("australian/J_bachelor/N", 2, CosSimilarity())