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
from composes.composition.dilation import Dilation
from composes.composition.full_additive import FullAdditive
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.utils import io_utils
from composes.utils import log_utils
import numpy as np

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno=0):
    print >>sys.stderr,\
    """Usage:
    python train_composition.py [options] [config_file]

    Options:
    -i --input <file>: input training file.
    -o --output <dir>: output directory. Resulted composition model is output 
                        in pickle format.
    -m --model <string>: one of WeightedAdditive/FullAdditive/LexicalFunction/
                        Dilation
    -r --regression <string>: one of lstsq/ridge. Optional, default lstsq.
    --crossvalidation <bool>: for -r=ridge, one of True/False. Optional, default True. 
    --intercept <bool>: one of True/False, default True.
    --param <scalar>: for -r=ridge, lambda value 
    --param_range <list(scalar)>: comma separated list of lambda values to be 
                                searched through when ridge regression uses 
                                crossvalidation. Optional, default linspace(0,0.5,10)
    -a --arg_space <file[,file]>: file(s) of argument semantic space (in pickle 
                        format). The second word of a word pair is interpreted 
                        in the second space, if provided.  
    -p --phrase_space <file>: file of phrase semantic space (in pickle format)
    --export_params: <bool>: if True, learned parameters are exported in an 
                    appropriate format.
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
        
def assert_bool(option, message):
    if option not in (True, False):
        print message
        usage(1)        

def train_model(in_file, out_dir, model, arg_space_files, phrase_space_file, regression, 
                crossvalid, intercept, param, param_range, export_params):
    
    print "Reading in data..."
    in_descr = in_file.split("/")[-1] 
    
    model_dict = {"weighted_add": WeightedAdditive,
                  "full_add": FullAdditive,
                  "lexical_func": LexicalFunction,
                  "dilation": Dilation
                  }
    learner_dict = {"ridge": RidgeRegressionLearner,
                    "lstsq": LstsqRegressionLearner
                    }
    
    arg_space = io_utils.load(arg_space_files[0], Space)
    arg_space2 = None
    if len(arg_space_files) == 2:
        arg_space2 = io_utils.load(arg_space_files[1], Space)
    
    phrase_space = io_utils.load(phrase_space_file, Space)    
        
    if not model in model_dict:
        raise ValueError("Invalid model:%s for training" % model)    
    
    model_cls = model_dict[model]
    if model_cls in (WeightedAdditive, Dilation):
        model_obj = model_cls()
    else:
        if regression in learner_dict:
            regression_obj = learner_dict[regression](crossvalidation=crossvalid,
                                                       intercept=intercept,
                                                       param=param,
                                                       param_range=param_range)
            model_obj = model_cls(learner=regression_obj)
            
            
    train_data = io_utils.read_tuple_list(in_file, 3)
    
    print "Training %s model" % model
    if arg_space2 is None or model == "lexical_func":
        model_obj.train(train_data, arg_space, phrase_space)
    else:
        model_obj.train(train_data, (arg_space, arg_space2), phrase_space)
    
    print "Printing..."    
    out_file = ".".join([out_dir + "TRAINED_COMP_MODEL", model, in_descr])    
    io_utils.save(model_obj, "%s.pickle" % out_file)
    
    if export_params:
        model_obj.export("%s.params" % out_file)
        
    
def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:m:r:a:p:l:", 
                                   ["help", "input=", "output=", "model=",
                                    "regression=", "intercept=", "arg_space=",
                                    "phrase_space=", "export_params=", "log=",
                                    "crossvalidation=", "param="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

       
    out_dir = None
    in_file = None
    model = None
    regression = None
    crossvalidation = False
    intercept = True
    param_range = None
    arg_space = None
    phrase_space = None
    export_params= False
    log_file = None
    param = None 
          
    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        out_dir = config.get("output") if config.has_option("output") else None
        in_file = config.get("input") if config.has_option("input") else None
        model = config.get("model") if config.has_option("model") else None
        regression = config.get("regression") if config.has_option("regression") else None
        crossvalidation = config.get("crossvalidation") if config.has_option("crossvalidation") else False
        intercept = config.get("intercept") if config.has_option("intercept") else True
        param_range = config.get("param_range") if config.has_option("param_range") else None
        param = config.get("param") if config.has_option("param") else None
        arg_space = config.get("arg_space") if config.has_option("arg_space") else None
        phrase_space = config.get("phrase_space") if config.has_option("phrase_space") else None
        export_params = config.get("export_params") if config.has_option("export_params") else False
        log_file = config.get("log") if config.has_option("log") else None
            
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-m", "--model"):
            model = val
        elif opt in ("-a", "--arg_space"):
            arg_space = val.split(",")
        elif opt in ("-p", "--phrase_space"):
            phrase_space = val
        elif opt in ("-r", "--regression"):
            regression = val
        elif opt in ("--crossvalidation"):
            crossvalidation = eval(val)
        elif opt in ("--intercept"):
            intercept = eval(val)
        elif opt in ("--param"):
            print opt, val
            param = float(val)
        elif opt in ("--param_range"):
            param_range = val.split(",")
            param_range = [float(param) for param in param_range]
        elif opt in ("--export_params"):
            export_params = eval(val)
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
    assert_option_not_none(model, "Model to be trained required")
    assert_option_not_none(arg_space, "Argument space(s) file(s) required")
    assert_option_not_none(phrase_space, "Phrase space file required")
    assert_bool(intercept, "intercept must be True/False")
    assert_bool(crossvalidation, "crossvalidation must be True/False")
    assert_bool(export_params, "export_params must be True/False")
    
    if not crossvalidation and regression == "ridge":
        assert_option_not_none(param, "Cannot run (no-crossvalidation) RidgeRegression with no lambda value!")
        
    train_model(in_file, out_dir, model, arg_space, phrase_space, regression, 
                crossvalidation, intercept, param, param_range, export_params)
    
    
if __name__ == '__main__':
    main(sys.argv)    