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
import pipeline_utils as utils 

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
    -c --core <file>: core space file (pickle dump of a space, .pkl expected)
    --core_in_dir: <dir>: input directory, all files that pass the --filter are tested.
                -i value is ignored. Optional.
    --core_filter: <string>: when --core_in_dir, it acts as a filter on the files to be tested:
                only files containing this substring are tested. Optional, 
                default all files in in_dir are tested.
    -l --log <file>: log file. Optional, default ./build_core_space.log
    --input_format: <string>: one of sm(sparse matrix), dm(dense matrix), pkl(pickle). 
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


def build_space(in_file_prefix, in_format, out_dir, out_format, core_space_file,
                is_gz):

    core_space = io_utils.load(core_space_file, Space)
    in_file_descr = "PER_SS." + in_file_prefix.split("/")[-1]
    core_descr = ".".join(core_space_file.split("/")[-1].split(".")[0:-1])
     
    if not in_format in ("sm", "dm", "pkl"):
        raise ValueError("Invalid input format:%s" % in_format) 
    
    data_file = '%s.%s' % (in_file_prefix, in_format)
        
    if in_format == "pkl":
        space = io_utils.load(data_file, Space)
    else:
        if is_gz:
            data_file = '%s.gz' % data_file
        row_file = '%s.rows' % (in_file_prefix)
        column_file = '%s.cols' % (in_file_prefix)
        if not os.path.exists(row_file):
            row_file = None
        if not os.path.exists(column_file):
            column_file = None
            if in_format == "sm":
                raise ValueError("Column file: %s needs to be provided!" % column_file)
        print "Building matrix..."
        space = PeripheralSpace.build(core_space, data=data_file, rows=row_file, 
                                      cols=column_file, format=in_format)
    
    print "Printing..."
    out_file_prefix = "%s/%s.%s" % (out_dir, in_file_descr, core_descr)
    io_utils.save(space, out_file_prefix + ".pkl")
    if not out_format is None:
        space.export(out_file_prefix, format=out_format)  

def build_space_batch(in_file_prefix, in_format, out_dir, out_format, 
                      core_in_dir, core_filter, is_gz):

    if not os.path.exists(core_in_dir):
        raise ValueError("Input directory not found: %s" % core_in_dir)
    
    if not core_in_dir.endswith("/"):
        core_in_dir = core_in_dir + "/"
        
    for file_ in os.listdir(core_in_dir):
        if file_.find(core_filter) != -1 and file_.endswith(".pkl"):
            print file_
            core_space_file = core_in_dir + file_ 
            build_space(in_file_prefix, in_format, out_dir, out_format, 
                        core_space_file, is_gz)
    
    
def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:c:l:", 
                                   ["help", "input=", "output=", "core=", 
                                    "log=", "input_format=", "output_format=",
                                    "core_in_dir=", "core_filter=", "gz="])
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
    core_in_dir = None
    core_filter = ""
    gz = "False"
    
    section = "build_peripheral_space"
    
    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        out_dir = utils.config_get(section, config, "output", None) 
        in_file_prefix = utils.config_get(section, config, "input", None)
        core_space_file = utils.config_get(section, config, "core", None) 
        core_in_dir = utils.config_get(section, config, "core_in_dir", None) 
        core_filter = utils.config_get(section, config, "core_filter", "") 
        log_file = utils.config_get(section, config, "log", None) 
        in_format = utils.config_get(section, config, "input_format", None) 
        out_format = utils.config_get(section, config, "output_format", None)
        gz = utils.config_get(section, config, "gz", gz) 
            
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file_prefix = val 
        elif opt in ("-o", "--output"):
            out_dir = val
        elif opt == "--gz":
            gz = val  
        elif opt in ("-c", "--core"):
            core_space_file = val 
        elif opt in ("-l", "--log"):
            log_file = val 
        elif opt == "--input_format":
            in_format = val 
        elif opt == "--output_format":
            out_format = val 
        elif opt == "--core_in_dir":
            core_in_dir = val    
        elif opt == "--core_filter":
            core_filter = val    
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            usage(1)
            
    log_utils.config_logging(log_file)

    utils.assert_option_not_none(in_file_prefix, "Input file prefix required", usage)
    utils.assert_option_not_none(out_dir, "Output directory required", usage)    
    utils.assert_option_not_none(in_format, "Input file format required", usage)
    
    gz = eval(gz)
    utils.assert_bool(gz, "--gz value must be True/False", usage)
    
    if not core_in_dir is None:
        build_space_batch(in_file_prefix, in_format, out_dir, out_format, core_in_dir, core_filter, gz)
    else:
        utils.assert_option_not_none(core_space_file, "Input file required", usage)
        build_space(in_file_prefix, in_format, out_dir, out_format, core_space_file, gz)
   
if __name__ == '__main__':
    main(sys.argv)    
