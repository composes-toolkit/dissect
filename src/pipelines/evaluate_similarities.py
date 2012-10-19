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
from composes.utils import scoring_utils
from composes.utils import log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno=0):
    print >>sys.stderr,\
    """Usage:
    python compute_similarities.py [options] [config_file]

    Options:
    -i --input <file>: input file.
    -m --correlation_measures <list(string)>: comma-separated correlation measures
    -c --columns <(int,int)>: pair of columns, indicating which columns contain 
            the words to be compared
    -l --log <file>: log file. Optional, default ./build_core_space.log
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
        

def evaluate_sim(in_file, columns, corr_measures):
    
    if not len(columns) == 2:
        raise ValueError("Column description unrecognized!") 
    
    gold = []
    prediction = []
    with open(in_file) as in_stream:
        for line in in_stream:
            if not line.strip() == "":
                elems = line.strip().split()
                gold.append(elems[columns[0]])
                prediction.append(elems[columns[1]])
    
    for corr_measure in corr_measures:
        print "CORELATION:s" % corr_measure                    
        corr = scoring_utils.score(gold, prediction, corr_measure)
        print "\t%f" % corr                    


def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:m:c:l:", 
                                   ["help", "input=", "correlation_measures=",
                                    "columns=", "log=" ])
        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    in_file = None
    corr_measures = None
    columns = None
    log_file = None

    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        in_file = config.get("input") if config.has_option("input") else None
        corr_measures = config.get("correlation_measures").split(",") if config.has_option("correlation_measures") else None
        columns = config.get("columns").split(",") if config.has_option("columns") else None
        log_file = config.get("log") if config.has_option("log") else None
    
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-m", "--correlation_measures"):
            corr_measures = val.split(",") 
        elif opt in ("-c", "--columns"):
            columns = val.split(",") 
        elif opt in ("-l", "--log"):
            log_file = val 
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            usage(1)
            
    log_utils.config_logging(log_file)

    assert_option_not_none(in_file, "Input file required")
    assert_option_not_none(corr_measures, "Correlation measures required")
    assert_option_not_none(columns, "Columns to be read from input file required")
        
    evaluate_sim(in_file, columns, corr_measures)
    
   
if __name__ == '__main__':
    main(sys.argv)
    