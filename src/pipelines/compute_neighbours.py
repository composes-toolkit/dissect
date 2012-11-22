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
import pipeline_utils as utils
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
       

def compute_neighbours(in_file, no_neighbours, out_dir, sim_measure, space_files):
    sim_dict = {"cos": CosSimilarity(),
                "lin": LinSimilarity(),
                "dot_prod": DotProdSimilarity(),
                "euclidean": EuclideanSimilarity()}
    
    if not sim_measure in sim_dict:
        raise ValueError("Similarity measure:%s not defined" % sim_measure)
    
    space = io_utils.load(space_files[0], Space)
    space2 = None
    space_descr = ".".join(space_files[0].split("/")[-1].split(".")[0:-1])
    if len(space_files) == 2:
        space2 = io_utils.load(space_files[1], Space)
        space_descr = ".".join([space_descr] + space_files[1].split("/")[-1].split(".")[0:-1])
        
    sim = sim_dict[sim_measure]
    
    descr = ".".join(["NEIGHBOURS", in_file.split("/")[-1], space_descr])
    out_file = '%s/%s.%s' % (out_dir, descr, sim_measure)
    io_utils.create_parent_directories(out_file)
        
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
    no_neighbours = "20"
    

    if (len(argv) == 1):
        config_file = argv[0]
        with open(config_file) as f:
            pass
        config = ConfigParser()
        config.read(config_file)
        out_dir = utils.config_get(section, config, "output", None) 
        in_file = utils.config_get(section, config, "input", None)
        sim_measure = utils.config_get(section, config, "sim_measure", None) 
        spaces = utils.config_get(section, config, "space", None)
        if not spaces is None:
            spaces = spaces.split(",")
        no_neighbours = utils.config_get(section, config, "no_neighbours", no_neighbours) 
        log_file = utils.config_get(section, config, "log", None) 
        
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

    no_neighbours = int(no_neighbours)
    
    utils.assert_option_not_none(in_file, "Input file required", usage)
    utils.assert_option_not_none(out_dir, "Output directory required", usage)    
    utils.assert_option_not_none(sim_measure, "Similarity measure required", usage)
    utils.assert_option_not_none(spaces, "Semantic space file required", usage)
        
    compute_neighbours(in_file, no_neighbours, out_dir, sim_measure, spaces)
    
    
   
if __name__ == '__main__':
    main(sys.argv)