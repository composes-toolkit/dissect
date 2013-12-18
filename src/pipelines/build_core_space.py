'''
Created on Oct 17, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
from composes.transformation.external.matlab_nmf import MatlabNmf

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
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
from composes.transformation.scaling.plog_weighting import PlogWeighting
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection
from composes.transformation.dim_reduction.nmf import Nmf
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.scaling.normalization import Normalization
from composes.transformation.scaling.row_normalization import RowNormalization
from composes.utils import io_utils
from composes.utils import log_utils
import pipeline_utils as utils

import logging
logger = logging.getLogger("test vector space construction pipeline")



def usage(errno=0):
    print >>sys.stderr,\
    """
    Usage:
    python build_core_space.py [options] [config_file]
    \n\
    Options:
    -i --input <file>: prefix of input files.
    -o --output <dir>: output directory. Space is output in pickle format.
    -w --weighting <list(string)>: comma-separated weighting schemes. Optional.
    -s --selection <list(string_string_int)>: comma-separated feature selection methods.
            A feature selection method is: method_criterion_reduced-dim. Method is
            one of "top", criterion one of "sum/length". Examples: top_sum_1000,
            top_length_1000. Optional.
    -r --reduction <list(string_int)>: comma-separated dimension reduction types.
             A reduction is: method_reduced-dim. Method is one of "svd/nmf".
             Examples: svd_300, nmf_100. Optional.
    -n --normalization <list(strings)>: comma-separated normalization methods,
         one of row/all. Row normalizes rows to length 1, sum normalizes sum
         of all elements 1.
    -l --log <file>: log file. Optional.
    --input_format: <string>: one of sm(sparse matrix), dm(dense matrix), pkl (pickle).
    --gz <bool>: if --input_format=sm, True if the input matrix file is zipped.
            Option, default False.
    --output_format: <string> Additional output format: one of sm(sparse matrix),
            dm(dense matrix). Optional.
    -h --help : help

    Arguments:
    config_file: <file>, used as default values for configuration options above.
            If you don't specify these options in [options] the value from the
            config_file will be used. Optional.

    Example:
    """
    sys.exit(errno)


def apply_weighting(space, w):

    weightings_dict = {"plog":PlogWeighting(),
                      "ppmi":PpmiWeighting(),
                      "epmi":EpmiWeighting(),
                      "plmi":PlmiWeighting()}

    if not w in (None, "none"):
        print "Applying weighting: %s" % w
        if not w in weightings_dict:
            warn("Weigthing scheme: %s not defined" % w)
            return space

        w_space = space.apply(weightings_dict[w])
    else:
        w_space = space

    return w_space


def apply_selection(w_space, s):

    selections_dict = {"top": TopFeatureSelection}
    selection_crit_list = ["sum", "length"]

    if not s in (None, "none"):
        print "Applying feature selection: %s" % s
        sel_els = s.split("_")
        if not len(sel_els) == 3:
            warn("Feature selection: %s not defined" % s)
            return w_space
        [s_type, s_criterion, s_red_dim] = sel_els
        if not s_type in selections_dict:
            warn("Feature selection: %s not defined" % s)
            return w_space
        if not s_criterion in selection_crit_list:
            warn("Feature selection: %s not defined" % s)
            return w_space

        sel_cls = selections_dict[s_type]
        feat_sel = sel_cls(int(s_red_dim),criterion=s_criterion)
        s_space = w_space.apply(feat_sel)

    else:
        s_space = w_space

    return s_space

def apply_reduction(s_space, r):
    reductions_dict = {"svd": Svd,
                       "nmf": Nmf,
                       "mlabnmf": MatlabNmf}
        
    if not r in (None, "none"):
        print "Applying dimensionality reduction: %s" % r
        red_els = r.split("_")
        if not len(red_els) == 2:
            warn("Dimensionality reduction: %s not defined" % r)
            return s_space
        [r_type, r_dim_red] = red_els
        if not r_type in reductions_dict:
            warn("Dimensionality reduction: %s not defined" % r)
            return s_space

        red_cls = reductions_dict[r_type]
        red = red_cls(int(r_dim_red))
        r_space = s_space.apply(red)
    else:
        r_space = s_space

    return r_space

def apply_normalization(r_space, n):
    normalizations_dict = {"all": Normalization,
                           "row": RowNormalization}

    if not n in (None, "none"):
        print "Applying normalization: %s" % n
        if not n in normalizations_dict:
            warn("Normalization: %s not defined" % n)
            return r_space

        norm_cls = normalizations_dict[n]
        norm = norm_cls()
        n_space = r_space.apply(norm)
    else:
        n_space = r_space

    return n_space


def print_space(space, out_dir, op_list, out_format):

    ops = [op for op in op_list if (op and (not op == "none"))]
    space_descr = ".".join(ops)
    out_file = out_dir + "/" + space_descr

    io_utils.save(space, out_file + ".pkl")
    if not out_format is None:
        space.export(out_file, format=out_format)


def build_spaces(in_file_prefix, in_format, out_dir, out_format, weightings,
                 selections, reductions, normalizations, is_gz):

    in_file_descr = "CORE_SS." + in_file_prefix.split("/")[-1]
    data_file = '%s.%s' % (in_file_prefix, in_format)

    if not in_format in ("sm", "dm", "pkl"):
        raise ValueError("Invalid input format:%s" % in_format)

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
            if in_format == "sm":
                raise ValueError("Column file: %s needs to be provided!"
                                 % column_file)
            column_file = None

        print "Building matrix..."
        space = Space.build(data=data_file, rows=row_file, cols=column_file,
                            format=in_format)

    for w in weightings:
        w_space = apply_weighting(space, w)

        for s in selections:
            s_space = apply_selection(w_space, s)

            for r in reductions:
                r_space = apply_reduction(s_space, r)

                for n in normalizations:
                    n_space = apply_normalization(r_space, n)

                    print "Printing..."
                    print_space(n_space, out_dir, [in_file_descr, w, s, r, n], out_format)


def main(sys_argv):

    try:
        opts, argv = getopt.getopt(sys_argv[1:], "hi:o:w:s:r:n:l:",
                                   ["help", "input=", "output=",
                                    "weighting=", "selection=", "reduction=",
                                     "normalization=", "log=", "gz=",
                                     "input_format=", "output_format="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    out_dir = None
    in_file_prefix = None
    weightings = [None]
    selections = [None]
    reductions = [None]
    normalizations = [None]
    log_file = None
    in_format = None
    out_format = None
    gz = "False"
    section = "build_core_space"

    if len(argv) == 1:
        config_file = argv[0]
        with open(config_file) as f:
            pass
        config = ConfigParser()
        config.read(config_file)
        out_dir = utils.config_get(section, config, "output", None)
        in_file_prefix = utils.config_get(section, config, "input", None)
        weightings = utils.config_get(section, config, "weighting", [None])
        if not weightings == [None]:
            weightings = weightings.split(",")

        selections = utils.config_get(section, config, "selection", [None])
        if not selections == [None]:
            selections = selections.split(",")

        reductions = utils.config_get(section, config, "reduction", [None])
        if not reductions == [None]:
            reductions = reductions.split(",")

        normalizations = utils.config_get(section, config, "normalization", [None])
        if not normalizations == [None]:
            normalizations = normalizations.split(",")

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
        elif opt in ("-w", "--weighting"):
            weightings = val.split(",")
        elif opt in ("-s", "--selection"):
            selections = val.split(",")
        elif opt in ("-r", "--reduction"):
            reductions = val.split(",")
        elif opt in ("-n", "--normalization"):
            normalizations = val.split(",")
        elif opt in ("-l", "--log"):
            log_file = val
        elif opt == "--input_format":
            in_format = val
        elif opt == "--output_format":
            out_format = val
        elif opt in ("-h", "--help"):
            usage(0)
        else:
            usage(1)

    if not log_file is None:
        log_utils.config_logging(log_file)

    utils.assert_option_not_none(in_file_prefix, "Input file prefix required", usage)
    utils.assert_option_not_none(out_dir, "Output directory required", usage)
    utils.assert_option_not_none(in_format, "Input format required", usage)

    gz = eval(gz)

    utils.assert_bool(gz, "--gz value must be True/False", usage)

    build_spaces(in_file_prefix, in_format, out_dir, out_format, weightings,
                 selections, reductions, normalizations, gz)


if __name__ == '__main__':
    main(sys.argv)

