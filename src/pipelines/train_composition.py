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
import os
from warnings import warn
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



def usage(errno):
    print >>sys.stderr,\
    """Usage:
    python train_composition.py [options] [config_file]

    Options:
    -i --input <file>: input training file.
    -o --output <dir>: output directory. Resulted composition model is output 
                        in pickle format. Additionally, learned parameters are 
                        exported in appropriate format.
    -m --model <string>: one of WeightedAdditive/FullAdditive/LexicalFunction/
                        Dilation
    -r --regression <string>: one of lstsq/ridge. Optional, default lstsq.
    --crossvalidation <bool>: for -r=ridge, one of True/False. Optional, default True. 
    --intercept <bool>: one of True/False, default True
    -a --arg_space <file>: file of argument semantic space  
    -p --phrase_space <file>: file of phrase semantic space
    --output_format: <string> For LexicalFunction model, additional output 
            format: one of sm(sparse matrix), dm(dense matrix). Optional.
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
        

def compute_sim(in_file, columns, out_dir, sim_measures, space_files):
    
    sim_dict = {"cos": CosSimilarity(),
                "lin": LinSimilarity(),
                "dot_prod": DotProdSimilarity(),
                "euclidean": EuclideanSimilarity()}
    
    if not len(columns) == 2:
        raise ValueError("Column description unrecognized!") 
    
    in_descr = in_file.split("/")[-1] 
    
    space = io_utils.load(space_files[0], Space)
    space2 = None
    if len(space_files) == 2:
        space2 = io_utils.load(space_files[1], Space)
    
    for sim_measure in sim_measures:
        if not sim_measure in sim_dict:
            warn("Similarity measure:%s not defined" % sim_measure)
            continue
        
        sim = sim_dict[sim_measure]
        out_file = '%s/%s.%s' % (out_dir, in_descr, sim_measure)
        
        with open(in_file) as in_stream, open(out_file,"w") as out_stream:
            for line in in_stream:
                if not line.strip() == "":
                    elems = line.strip().split()
                    word1 = elems[columns[0]]
                    word2 = elems[columns[1]]
                 
                    predicted_sim = space.get_sim(word1, word2, sim, space2)
                    out_stream.write("%s %s\n" % (line.strip(), str(predicted_sim)))
 

def main():
    try:
        opts, argv = getopt.getopt(sys.argv[1:], "hi:o:m:r:a:p:l:", 
                                   ["help", "input=", "output=", "model=",
                                    "regression=", "intercept=", "arg_space=",
                                    "phrase_space=", "output_format=", "log=",])
          
        if (len(argv) == 1):
            config_file = argv[0]
            config = ConfigParser()
            config.read(config_file)
            out_dir = config.get("output") if config.has_option("output") else None
            in_file = config.get("input") if config.has_option("input") else None
            model = config.get("model") if config.has_option("model") else None
            regression = config.get("regression") if config.has_option("regression") else None
            crossvalidation = config.get("crossvalidation") if config.has_option("crossvalidation") else None
            intercept = config.get("intercept") if config.has_option("intercept") else None
            arg_space = config.get("arg_space") if config.has_option("arg_space") else None
            phrase_space = config.get("phrase_space") if config.has_option("phrase_space") else None
            output_format = config.get("output_format") if config.has_option("output_format") else None
            log_file = config.get("log") if config.has_option("log") else None
        else:
            usage(1)
                        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-m", "--model"):
            model = val
        elif opt in ("-a", "--arg_space"):
            arg_space_file = val
        elif opt in ("-p", "--phrase_space"):
            phrase_space_file = val
        elif opt in ("-r", "--regression"):
            regression = val
        elif opt in ("--crossvalidation"):
            crossvalidation = val
        elif opt in ("--intercept"):
            intercept = val
        elif opt in ("--output_format"):
            output_format = val
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
    assert_option_not_none(sim_measures, "Similarity measures required")
    assert_option_not_none(spaces, "Semantic space file required")
    assert_option_not_none(columns, "Columns to be read from input file required")
        
    compute_sim(in_file, columns, out_dir, sim_measures, spaces)
    
    
   
if __name__ == '__main__':
    main()