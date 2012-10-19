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
from ConfigParser import ConfigParser
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.utils import io_utils
from composes.utils import log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno=0):
    print >>sys.stderr,\
    """Usage:
    python build_peripheral_space.py [options] [config_file]
    \n\
    Options:
    -i --input <file>: prefix of input file.
    -o --output <dir>: output directory. Space is output in pickle format.
    -c --core <file>: core space file (pickle dump of a space, .pickle expected)
    -l --log <file>: log file. Optional, default ./build_core_space.log
    --input_format: <string>: one of sm(sparse matrix), dm(dense matrix), pickle. 
    --output_format: <string> Additional output format: one of sm(sparse matrix), 
            dm(dense matrix). Optional.
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

def build_space(in_file_prefix, in_format, out_dir, out_format, core_space_file):

    core_space = io_utils.load(core_space_file, Space)
    in_file_descr = "PER_SS." + in_file_prefix.split("/")[-1]
    core_descr = ".".join(core_space_file.split("/")[-1].split(".")[0:-1])
     
    if not in_format in ("sm, dm"):
        raise ValueError("Invalid input format:%s" % in_format) 
    data_file = '%s.%s' % (in_file_prefix, in_format)
    
    row_file = '%s.rows' % (in_file_prefix)
    column_file = '%s.cols' % (in_file_prefix)

    if not os.path.exists(row_file):
        row_file = None

    if not os.path.exists(column_file):
        column_file = None

    print "Building matrix..."
    space = PeripheralSpace.build(core_space, data=data_file, rows=row_file, 
                                  cols=column_file, format=in_format)
    
    print "Printing..."
    out_file_prefix = "%s/%s.%s" % (out_dir, in_file_descr, core_descr)
    io_utils.save(space, out_file_prefix + ".pickle")
    if not out_format is None:
        space.export(out_file_prefix, format=out_format)  

def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:c:l:", 
                                   ["help", "input=", "output=", "core=", 
                                    "log=", "input_format=", "output_format="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    out_dir = None
    in_file_prefix = None
    core_space_file = None
    log_file = None
    in_format = None
    out_format = None
    
    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        out_dir = config.get("output") if config.has_option("output") else None
        in_file_prefix = config.get("input") if config.has_option("input") else None
        core_space_file = config.get("core") if config.has_option("core") else None
        log_file = config.get("log") if config.has_option("log") else None
        in_format = config.get("input_format") if config.has_option("input_format") else None
        out_format = config.get("output_format") if config.has_option("output_format") else None
            
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file_prefix = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-c", "--core"):
            core_space_file = val 
        elif opt in ("-l", "--log"):
            log_file = val 
        elif opt in ("--input_format"):
            in_format = val 
        elif opt in ("--output_format"):
            out_format = val 
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            usage(1)
            
    log_utils.config_logging(log_file)

    assert_option_not_none(in_file_prefix, "Input file prefix required")
    assert_option_not_none(out_dir, "Output directory required")    
    assert_option_not_none(in_format, "Input file format required")
    assert_option_not_none(core_space_file, "Core space file required")
    
    build_space(in_file_prefix, in_format, out_dir, out_format, core_space_file)
   
if __name__ == '__main__':
    main(sys.argv)    
