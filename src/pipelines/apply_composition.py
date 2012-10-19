'''
Created on Oct 17, 2012

@author: georgianadinu
'''

import sys
import getopt
from ConfigParser import ConfigParser
from composes.semantic_space.space import Space
from composes.composition.composition_model import CompositionModel
from composes.composition.dilation import Dilation
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.lexical_function import LexicalFunction
from composes.composition.multiplicative import Multiplicative
from composes.utils import io_utils
from composes.utils import log_utils

import logging
logger = logging.getLogger("test vector space construction pipeline")

def usage(errno):
    print >>sys.stderr,\
    """Usage:
    python train_composition.py [options] [config_file]

    Options:
    -i --input <file>: input file.
    -o --output <dir>: output directory. Resulted composed space is output 
                        in pickle format.
    -m --model <string>: one of WeightedAdditive/FullAdditive/LexicalFunction/
                        Dilation/Multiplicative
    --alpha <scalar>: parameter for model=WeightedAdditive
    --beta <scalar>: parameter for model=WeightedAdditive
    --lambda <scalar>: parameter for model=Dilation
    -t --trained_model <file>: file containing a previously trained model.
    -a --arg_space <file[,file]>: file(s) of argument semantic space (in pickle 
                        format). The second word of a word pair is interpreted 
                        in the second space, if provided.
    --output_format: <string> Additional output format for the resulted composed 
                        space: one of sm(sparse matrix), dm(dense matrix). 
                        Optional.                          
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

def assert_xor_options(option1, option2, message):
    if not ((option1 is None) ^ (option2 is None)):
        print message
        usage(1) 

def create_model(model, alpha, beta, lambda_): 

    model_dict = {"weighted_add": WeightedAdditive,
                 "dilation": Dilation,
                 "mult": Multiplicative
                 }
    if not model in model_dict:
        raise ValueError("Invalid model:%s" % model)
    
    if model == "weighted_add":
        model_obj = WeightedAdditive(alpha, beta)    
                
    elif model == "dilation":
        model_obj = Dilation(lambda_)    
    
    return model_obj

    
def apply_model(in_file, out_dir, model, trained_model, arg_space_files,
                alpha, beta, lambda_, out_format):

    print "Reading in data..."
    in_descr = in_file.split("/")[-1] 
    
    if not model is None: 
        model_obj = create_model(model, alpha, beta, lambda_)
    else:
        model_obj = io_utils.load(trained_model, CompositionModel)
        
    model_descr = type(model_obj).__name__
     
    arg_space = io_utils.load(arg_space_files[0], Space)
    arg_space2 = None
    if len(arg_space_files) == 2:
        arg_space2 = io_utils.load(arg_space_files[1], Space)
    
    data = io_utils.read_tuple_list(in_file, 3)
    
    print "Applying composition model:%s" % type(model_obj)
    if arg_space2 is None or type(model_obj) in (LexicalFunction):
        composed_space = model_obj.compose(data, arg_space)
    else:
        composed_space = model_obj.compose(data, (arg_space, arg_space2))
    
    print "Printing..."
    out_file = ".".join(["COMPOSED_SPACE", model_descr, in_descr])    
    io_utils.save(composed_space, "%s.pickle" % out_file)
    
    if not out_format is None:
        out_file = "%s.%s" % (out_file, format)
        composed_space.export(out_file, format=out_format)
         
    
def main(sys_argv):
    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:m:a:t:l:", 
                                   ["help", "input=", "output=", "model=",
                                    "alpha=", "beta=", "lambda=", "arg_space=", 
                                    "trained_model=", "out_format=", "log="])
        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    out_dir = None
    in_file = None
    model = None
    arg_space = None
    trained_model = None
    alpha = None
    beta = None
    lambda_ = None
    log_file = None
    out_format = None 
          
    if (len(argv) == 1):
        config_file = argv[0]
        config = ConfigParser()
        config.read(config_file)
        out_dir = config.get("output") if config.has_option("output") else None
        in_file = config.get("input") if config.has_option("input") else None
        model = config.get("model") if config.has_option("model") else None
        trained_model = config.get("trained_model") if config.has_option("trained_model") else None
        arg_space = config.get("arg_space") if config.has_option("arg_space") else None
        alpha = config.get("alpha") if config.has_option("alpha") else None
        beta = config.get("beta") if config.has_option("beta") else None
        lambda_ = config.get("lambda_") if config.has_option("lambda_") else None
        log_file = config.get("log") if config.has_option("log") else None
        out_format = config.get("output_format") if config.has_option("output_format") else None

            
    for opt, val in opts:
        if opt in ("-i", "--input"):
            in_file = val 
        elif opt in ("-o", "--output"):
            out_dir = val 
        elif opt in ("-m", "--model"):
            model = val
        elif opt in ("-a", "--arg_space"):
            arg_space = val.split(",")
        elif opt in ("-t", "--trained_model"):
            trained_model = val
        elif opt in ("--alpha"):
            alpha = val 
        elif opt in ("--beta"):
            beta = val 
        elif opt in ("--lambda"):
            lambda_ = val 
        elif opt in ("--out_format"):
            out_format = val 
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
    assert_xor_options(model, trained_model, "(Only) one of model type (-m) or file of model object (-t) are required!")
    assert_option_not_none(arg_space, "Argument space(s) file(s) required")

        
    apply_model(in_file, out_dir, model, trained_model, arg_space,
                alpha, beta, lambda_, out_format)
    
    
if __name__ == '__main__':
    main(sys.argv)    
