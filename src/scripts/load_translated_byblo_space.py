import logging
from composes.utils import io_utils
from composes.composition.lexical_function import LexicalFunction
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from thesisgenerator.plugins.thesaurus_loader import Thesaurus


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


def train_baroni_composer(noun_events_file, ANs_events_file, output_prefix, row_transform=lambda x: x):
    logging.info('Starting training')
    logging.info('Nouns file is %s', noun_events_file)
    logging.info('ANs file is %s', ANs_events_file)

    # prepare the input files to be fed into Dissect
    #cleaned_an_file = '{}.uniq'.format(ANs_events_file)
    #noun_events_file = '{}.uniq'.format(noun_events_file)
    cleaned_nouns_file = _translate_byblo_to_dissect(noun_events_file, row_transform=row_transform)
    cleaned_an_file = _translate_byblo_to_dissect(ANs_events_file, row_transform=row_transform)

    #my_space = Thesaurus([noun_events_file], aggressive_lowercasing=False).to_dissect_core_space()
    my_space = Space.build(data="{}.sm".format(cleaned_nouns_file),
                           rows="{}.rows".format(cleaned_nouns_file),
                           cols="{}.cols".format(cleaned_nouns_file),
                           format="sm")
    logging.info('Each unigram vector has dimensionality %r', my_space.element_shape)
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
    logging.info('Each peripheral vector has dimensionality %r', my_per_space.element_shape)
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
        all_data.append((adj, noun, '%s_%s' % (adj, noun)))


    #train a lexical function model on the data
    composer = LexicalFunction()
    composer.train(all_data, my_space, my_per_space)
    available_adjs = composer.function_space.id2row
    model_file = output_prefix + '.model.pkl'
    io_utils.save(composer, model_file)
    logging.info('Saving trained composer to %s', model_file)

    # compose a set of predefined phrases
    composed_space = composer.compose(all_data, my_space)
    matrix = composed_space.cooccurrence_matrix.mat
    rows = composed_space.id2row
    cols = composed_space.id2column
    compsosed_data_file = output_prefix + '.composed.pkl'
    io_utils.save(composed_space, compsosed_data_file)
    logging.info('Saving composed training data to %s', compsosed_data_file)

    return model_file
    #similarity within the learned functional space
    #logging.info('Done')
    #
    ## test the learnt product
    ##return (composed_space.row2id, composed_space.cooccurrence_matrix)
    #logging.info(composer.function_space.get_sim("african_army", "african_army", CosSimilarity()))
    #
    ## this computes the sim by a brute force
    #composed_space.get_neighbours('african_army', 4, CosSimilarity(), space2=my_per_space)
    #composed_space.get_row('african/J_army/N').mat


def _append_unless_exists(string, suffix):
    return string if string.endswith(suffix) else '%s%s' % (string, suffix)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    noun_events_file = './data/in/julie.nouns50.vectors'
    ANs_events_file = './data/in/julie.ANs50.vectors'
    output_prefix = './data/out/trained_AN'
    # execute lines below once to convert Julie's format to dissect
    # make sure all unigrams and n-grams are in the same format, e.g. have a PoS tag, ANs are _-separated, etc
    #clean_entries_inplace(noun_events_file, lambda x: _append_unless_exists(x, '/N'))
    #clean_entries_inplace(ANs_events_file, lambda x: julie_transform(x, separator='_'))

    composer = train_baroni_composer(noun_events_file, ANs_events_file, output_prefix)