import logging
from composes.composition.lexical_function import LexicalFunction
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.similarity.cos import CosSimilarity
from composes.semantic_space.space import Space
from thesisgenerator.plugins.thesaurus_loader import Thesaurus


def _julie_transform(input, pos1='J', pos2='N', separator='_'):
    '''african/J:amod-HEAD:ancestry -> african_ancestry'''
    noun = input.split(':')[-1]
    adj = input.split('/')[0]
    return '{}/{}{}{}/{}'.format(adj, pos1, separator, noun, pos2)


def _translate_byblo_to_dissect(events_file, row_transform=lambda x: x):
    """
    Translated Byblo-made vectors file to dissect format in the absence of features/entries files
    :param events_file: path to byblo-made vectors
    :type events_file: str
    :return: prefix of dissect-compatible data files
    :rtype: str
    """
    # remove duplicate head noun vectors, converting to a dissect sparse matrix format
    t = Thesaurus([events_file], aggressive_lowercasing=False)
    output_file = '{}.uniq'.format(events_file)
    t.to_dissect_sparse_files(output_file, row_transform=row_transform)
    return output_file


def train_baroni_composer(noun_events_file, ANs_events_file, output_filename):
    logging.info('Starting')

    # prepare the input files to be fed into Dissect
    #cleaned_an_file = '{}.uniq'.format(ANs_events_file)
    #noun_events_file = '{}.uniq'.format(noun_events_file)
    cleaned_nouns_file = _translate_byblo_to_dissect(noun_events_file)
    cleaned_an_file = _translate_byblo_to_dissect(ANs_events_file)

    #my_space = Thesaurus([noun_events_file], aggressive_lowercasing=False).to_dissect_core_space()
    my_space = Space.build(data="{}.sm".format(cleaned_nouns_file),
                           rows="{}.rows".format(cleaned_nouns_file),
                           cols="{}.cols".format(cleaned_nouns_file),
                           format="sm")

    #Linalg._NMF_MAX_ITER = 2
    #my_space = my_space.apply(Nmf(10))

    #create a peripheral space
    my_per_space = PeripheralSpace.build(my_space,
                                         data="{}.sm".format(cleaned_an_file),
                                         rows="{}.rows".format(cleaned_an_file),
                                         # The columns of the peripheral space have to be identical to those
                                         # in the core space (including their order)!
                                         cols="{}.cols".format(cleaned_nouns_file),
                                         format="sm")

    #for a, b in product(my_space.row2id, my_space.row2id):
    #    sim = my_space.get_sim(a, b, lin)
    #    if sim > 0:
    #        logging.info('%s <> %s = %0.2f', a, b, sim)
    #
    #for i, noun in enumerate(my_space.row2id):
    #    logging.info(my_space.get_neighbours(noun, 2, lin))
    #    #logging.info(my_space.get_neighbours(noun, 2, lin, space2=my_per_space))



    #export the space in sparse format
    #my_space.export("./data/out/ex01", format="dm")
    #my_space = my_space.apply(PpmiWeighting())
    #my_space = my_space.apply(Svd(5))

    # use the model to compose words in my_space
    all_data = []
    for an in my_per_space._row2id:
        adj, noun = an.split('_')
        print adj
        all_data.append((adj, noun, '%s_%s' % (adj, noun)))
    from random import sample

    all_data = sample(all_data, 20)

    #train a lexical function model on the data
    my_comp = LexicalFunction()
    my_comp.train(all_data, my_space, my_per_space, n_jobs=-1)

    #similarity within the learned functional space
    composed_space = my_comp.compose(all_data, my_space)
    from composes.utils.io_utils import save

    save(composed_space, output_filename)
    logging.info('Done')

    # test the learnt product
    #return (composed_space.row2id, composed_space.cooccurrence_matrix)
    logging.info(my_comp.function_space.get_sim("african_army", "african_army", CosSimilarity()))

    # this computes the sim by a brute force
    composed_space.get_neighbours('african_army', 4, CosSimilarity(), space2=my_per_space)
    composed_space.get_row('african_army').mat


def _clean_entries_inplace(filename, function, separator='\t'):
    import tempfile, shutil

    shutil.copy(filename, filename + '.bak')
    tmp = tempfile.NamedTemporaryFile(delete=False)
    with open(filename) as infile, open(tmp.name, 'w') as outfile:
        for line in infile:
            fields = line.split(separator)
            fields[0] = function(fields[0])
            outfile.write(separator.join(fields))
            #outfile.write('\n')
    shutil.move(tmp.name, filename)


def _append_unless_exists(string, suffix):
    return string if string.endswith(suffix) else '%s%s' % (string, suffix)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    noun_events_file = './data/in/julie.nouns.vectors'
    ANs_events_file = './data/in/julie.ANs.vectors'
    output_filename = 'data/out/trained_AN.pkl'
    #_clean_entries_inplace(noun_events_file, lambda x: _append_unless_exists(x, '/N'))
    #_clean_entries_inplace(ANs_events_file, lambda x: _julie_transform(x, separator='_'))

    train_baroni_composer(noun_events_file, ANs_events_file, output_filename)