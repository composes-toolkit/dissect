'''
Created on Oct 17, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
'''
Created on Oct 17, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

'''
Created on Jun 12, 2012

@author: thenghia.pham
'''


import sys
import getopt
import os
from ConfigParser import ConfigParser
from composes.utils import scoring_utils
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
    --in_dir: <dir>: input directory, all files that pass the --filter are tested.
                -i value is ignored. Optional.
    --filter: <string>: when --in_dir, it acts as a filter on the files to be tested:
                only files containing this substring are tested. Optional, 
                default all files in in_dir are tested.
    -m --correlation_measure <list(string)>: comma-separated correlation measures
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

def evaluate_sim(in_file, columns, corr_measures):
    
    if not len(columns) == 2:
        raise ValueError("Column description unrecognized!") 
    col0 = int(columns[0]) - 1
    col1 = int(columns[1]) - 1
    
    gold = []
    prediction = []
    with open(in_file) as in_stream:
        for line in in_stream:
            if not line.strip() == "":
                elems = line.strip().split()
                gold.append(float(elems[col0]))
                prediction.append(float(elems[col1]))
    
    for corr_measure in corr_measures:
        print "CORRELATION:%s" % corr_measure                    
        corr = scoring_utils.score(gold, prediction, corr_measure)
        print "\t%f" % corr  

        
def evaluate_sim_batch(in_dir, columns, corr_measures, filter_=""):
    
    if not os.path.exists(in_dir):
        raise ValueError("Input directory not found: %s" % in_dir)
    
    if not in_dir.endswith("/"):
        in_dir = in_dir + "/"
        
    for file_ in os.listdir(in_dir):
        if file_.find(filter_) != -1:
            print file_
            evaluate_sim(in_dir + file_, columns, corr_measures)


def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:m:c:l:", 
                                   ["help", "input=", "correlation_measure=",
                                    "columns=", "log=", "in_dir=", "filter="])
        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    in_file = None
    in_dir = None
    filter_ = ""
    corr_measures = None
    columns = None
    log_file = None

    section = "evaluate_similarities"
    
    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        in_file = utils.config_get(section, config, "input", None) 
        in_dir = utils.config_get(section, config, "in_dir", None) 
        filter_ = utils.config_get(section, config, "filter", filter_) 
        corr_measures = utils.config_get(section, config, "correlation_measure", None) 
        if not corr_measures is None:
            corr_measures = corr_measures.split(",")
        columns = utils.config_get(section, config, "columns", None) 
        if not columns is None:
            columns = columns.split(",")
        log_file = utils.config_get(section, config, "log", None)  
        
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-m", "--correlation_measure"):
            corr_measures = val.split(",") 
        elif opt in ("-c", "--columns"):
            columns = val.split(",")
        elif opt == "--in_dir":
            in_dir = val
        elif opt == "--filter":
            filter_ = val    
        elif opt in ("-l", "--log"):
            log_file = val 
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            usage(1)
            
    log_utils.config_logging(log_file)
    
    utils.assert_option_not_none(corr_measures, "Correlation measures required", usage)
    utils.assert_option_not_none(columns, "Columns to be read from input file required", usage)
    
    if len(columns) != 2:
        raise ValueError("Columns (-c) field should contain two comma-separated integers (e.g. -c 3,4)")
            
    if not in_dir is None:
        evaluate_sim_batch(in_dir, columns, corr_measures, filter_)
    else:
        utils.assert_option_not_none(in_file, "Input file required", usage)
        evaluate_sim(in_file, columns, corr_measures)
   
if __name__ == '__main__':
    main(sys.argv)
    