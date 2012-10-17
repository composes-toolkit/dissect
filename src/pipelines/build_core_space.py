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
from composes.transformation.scaling.raw_weighting import RawWeighting
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
from composes.transformation.scaling.plog_weighting import PlogWeighting
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection
from composes.transformation.dim_reduction.nmf import Nmf
from composes.transformation.dim_reduction.svd import Svd
from composes.utils import io_utils
from composes.utils import log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno):
    print >>sys.stderr,\
    """Usage:
    python build_core_space.py [options] [config_file]
    \n\
    Options:
    -i --input <file>: prefix of input files.
    -o --output <dir>: output directory. Space is output in pickle format.
    -w --weighting <list(string)>: comma-separated weighting schemes. Optional.
    -s --selection <list(string_int)>: comma-separated feature selection methods 
            (e.g. top_sum_1000/top_length_1000). Optional.
    -r --reduction <list(string_int)>: comma-separated dimension reduction types 
            (e.g. svd_300, nmf_100). Optional.
    -l --log <file>: log file. Optional.
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
        
def apply_weighting(space, w):
    
    weightings_dict = {"raw":RawWeighting(),
                      "plog":PlogWeighting(),
                      "ppmi":PpmiWeighting(),
                      "epmi":EpmiWeighting(), 
                      "plmi":PlmiWeighting()}

    if not w is None:
        if not w in weightings_dict:
            warn("Weigthing scheme: %s not defined" % w)
            continue
        
        w_space = space.apply(weightings_dict[w])
    else:
        w_space = space
    
    return w_space    
        
        
def apply_selection(w_space, s):
    
    selections_dict = {"top": TopFeatureSelection}
    selection_crit_list = ["sum", "length"]
        
    if not s is None:
        sel_els = s.split("_")
        if not len(sel_els) == 3:
            warn("Feature selection: %s not defined" % s)
            continue
        [s_type, s_criterion, s_red_dim] = sel_els
        if not s_type in selections_dict:
            warn("Feature selection: %s not defined" % s)
            continue
        if not s_criterion in selection_crit_list:
            warn("Feature selection: %s not defined" % s)
            continue
        
        sel_cls = selections_dict[s_type]
        feat_sel = sel_cls(int(s_red_dim),criterion=s_criterion)
        
        s_space = w_space.apply(feat_sel)
    else:
        s_space = w_space    
       
    return s_space      
          
def apply_reduction(s_space, r):
    reductions_dict = {"svd": Svd,
                       "nmf": Nmf}
        
    if not r is None:
        red_els = r.split("_")
        if not len(red_els) == 2:
            warn("Dimensionality reduction: %s not defined" % r)
            continue
        [r_type, r_dim_red] = red_els
        if not r_type in reductions_dict:
            warn("Dimensionality reduction: %s not defined" % r)
            continue
        
        red_cls = reductions_dict[r_type]
        red = red_cls(int(r_dim_red))
        r_space = s_space.apply(red)
    else:
        r_space = s_space
      
    return r_space            
                
def build_spaces(in_file_prefix, in_format, out_dir, out_format, weightings, 
                 selections, reductions):

    in_file_descr = in_file_prefix.split("/")[-1]
    
    if (in_format == "sm"):
        data_file = '%s.sm' % (in_file_prefix)
    elif(in_format == "dm"):
        data_file = '%s.dm' % (in_file_prefix)
    
    row_file = '%s.rows' % (in_file_prefix)
    column_file = '%s.cols' % (in_file_prefix)

    if not os.path.exists(row_file):
        row_file = None

    if not os.path.exists(column_file):
        column_file = None

    space = Space.build(data=data_file, rows=row_file, cols=column_file, 
                        format=in_format)
    for w in weightings:
        w_space = apply_weighting(space, w)
        for s in selections:
            s_space = apply_selection(w_space, s)
            for r in reductions:
                r_space = apply_reduction(s_space, r)
                
                ops = [op for op in [in_file_descr, w, s, r] if op]     
                r_space_descr = ".".join(ops)
                out_file = out_dir + "/" + r_space_descr
                io_utils.save(r_space, out_file + ".pickle")
                if not out_format is None:
                    r_space.export(out_file, format=out_format)    


def main():
    try:
        opts, argv = getopt.getopt(sys.argv[1:], "hi:o:w:s:r:l:", 
                                   ["help", "input=", "output=", "weighting=",
                                    "selection=", "reduction=", "log=",
                                    "input_format=", "output_format="])
        if (len(argv) == 1):
            config_file = argv[0]
            config = ConfigParser()
            config.read(config_file)
            out_dir = config.get("output") if config.has_option("output") else None
            in_file_prefix = config.get("input") if config.has_option("input") else None
            weightings = config.get("weighting").split(",") if config.has_option("weighting") else [None]
            selections = config.get("selection").split(",") if config.has_option("selection") else [None]
            reductions = config.get("reduction").split(",") if config.has_option("reduction") else [None]
            log_file = config.get("log") if config.has_option("log") else None
            in_format = config.get("input_format") if config.has_option("input_format") else None
            out_format = config.get("output_format") if config.has_option("output_format") else None
        else:
            usage(1)
                        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file_prefix = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-w", "--weighting"):
            weightings = val.split(",") 
        elif opt in ("-s", "--selection"):
            selections = val.split(",") 
        elif opt in ("-r", "--reduction"):
            reductions = val.split(",") 
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
    assert_option_not_none(in_format, "Input format required")
        
    build_spaces(in_file_prefix, in_format, out_dir, out_format, weightings, 
                 selections, reductions)
    
   
if __name__ == '__main__':
    main()