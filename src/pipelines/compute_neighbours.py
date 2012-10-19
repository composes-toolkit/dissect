'''
Created on Oct 17, 2012

@author: georgianadinu
'''
'''
Created on Oct 17, 2012

@author: georgianadinu
'''

'''
Created on Jun 12, 2012

@author: thenghia.pham
'''


import sys
import getopt
from ConfigParser import ConfigParser
from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity
from composes.similarity.lin import LinSimilarity
from composes.similarity.dot_prod import DotProdSimilarity
from composes.similarity.euclidean import EuclideanSimilarity
from composes.utils import io_utils
from composes.utils import log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno=0):
    print >>sys.stderr,\
    """Usage:
    python compute_similarities.py [options] [config_file]

    Options:
    -i --input <file>: input file.
    -o --output <dir>: output directory. 
    -s --space <file[,file2]>: file of semantic space. The second 
            word of a word pair is interpreted in the second space argument, 
            if provided. 
    -m --sim_measure <string>: similarity measure
    -n --no_neighbours <int>: number of neighbours to be returned
    -l --log <file>: log file. Optional.
    -h --help : help
    
    Arguments:
    config_file: <file>, used as default values for configuration options above.
            If you don't specify these options in [options] the value from the 
            config_file will be used.
    
    Example:
    """
    sys.exit(errno)

def assert_option_not_none(option, message):
    if option is None:
        print message
        usage(1)
        

def compute_neighbours(in_file, no_neighbours, out_dir, sim_measure, space_files):
    no_neighbours = int(no_neighbours)
    sim_dict = {"cos": CosSimilarity(),
                "lin": LinSimilarity(),
                "dot_prod": DotProdSimilarity(),
                "euclidean": EuclideanSimilarity()}
    
    in_descr = "NEIGHBOURS." + in_file.split("/")[-1] 
    
    if not sim_measure in sim_dict:
        raise ValueError("Similarity measure:%s not defined" % sim_measure)
    
    space = io_utils.load(space_files[0], Space)
    space2 = None
    if len(space_files) == 2:
        space2 = io_utils.load(space_files[1], Space)
    
    sim = sim_dict[sim_measure]
    out_file = '%s/%s.%s' % (out_dir, in_descr, sim_measure)
    
    data = io_utils.read_list(in_file)

    print "Computing neighbours: %s" % sim_measure 
    with open(out_file,"w") as out_stream:
        for word in data:
            out_stream.write("%s\n" % word)
            result = space.get_neighbours(word, no_neighbours, sim, space2)
            for neighbour, neighbour_sim in result:
                out_stream.write("\t%s %s\n" % (neighbour, neighbour_sim)) 

def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:s:m:n:l:", 
                                   ["help", "input=", "output=", "sim_measures=",
                                    "space=", "log=", "no_neighbours="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    section = "compute_neighbours"

    out_dir = None
    in_file = None
    sim_measure = None
    spaces = None
    log_file = None
    no_neighbours = 20
    

    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        out_dir = config.get(section, "output") if config.has_option(section, "output") else None
        in_file = config.get(section, "input") if config.has_option(section, "input") else None
        sim_measure = config.get(section, "sim_measure") if config.has_option(section, "sim_measure") else None
        spaces = config.get(section, "space").split(",") if config.has_option(section, "space") else None
        no_neighbours = config.get(section, "no_neighbours").split(",") if config.has_option(section, "no_neighbours") else no_neighbours
        log_file = config.get(section, "log") if config.has_option(section, "log") else None
    
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-m", "--sim_measure"):
            sim_measure = val
        elif opt in ("-s", "--space"):
            spaces = val.split(",") 
        elif opt in ("-n", "--no_neighbours"):
            no_neighbours = val 
        elif opt in ("-l", "--log"):
            log_file = val 
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            usage(1)
            
    log_utils.config_logging(log_file)

    assert_option_not_none(in_file, "Input file required")
    assert_option_not_none(out_dir, "Output directory required")    
    assert_option_not_none(sim_measure, "Similarity measure required")
    assert_option_not_none(spaces, "Semantic space file required")
        
    compute_neighbours(in_file, no_neighbours, out_dir, sim_measure, spaces)
    
    
   
if __name__ == '__main__':
    main(sys.argv)