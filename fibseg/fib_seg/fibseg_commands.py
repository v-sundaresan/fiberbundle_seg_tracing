from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from fibseg.fib_seg import fibseg_test_function, fibseg_get_fiber_specs
import glob

#=========================================================================================
# Truenet commands function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================

##########################################################################################
# Define the evaluate sub-command for truenet
##########################################################################################


def evaluate(args):
    """
    :param args: Input arguments from argparse
    """
    # Do basic sanity checks and assign variable names
    inp_dir = args.inp_dir
    out_dir = args.output_dir

    if not os.path.isdir(inp_dir):
        raise ValueError(inp_dir + ' does not appear to be a valid input directory')

    input_sect_paths = glob.glob(os.path.join(inp_dir, '*_s*.jpg'))

    if len(input_sect_paths) == 0:
        raise ValueError(inp_dir + ' does not contain any histology sections/ filenames NOT in required format')

    if os.path.isdir(out_dir) is False:
        raise ValueError(out_dir + ' does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the test subjects
    sect_name_dicts = []
    for l in range(len(input_sect_paths)):
        basepath = input_sect_paths[l].split("_s")[0]
        basename = basepath.split(os.sep)[0]

        subj_name_dict = {'sect_path': input_sect_paths[l],
                          'basename': basename}
        sect_name_dicts.append(subj_name_dict)

    if args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    else:
        if os.path.isfile(args.model_name) is False:
            raise ValueError('In directory ' + os.path.dirname(args.model_name) +
                             'does not appear to be a valid model file')
        else:
            model_dir = os.path.dirname(args.model_name)
            model_name = os.path.basename(args.model_name)

    # Create the training parameters dictionary
    eval_params = {'Nclass': args.num_classes,
                   'EveryN': args.cp_everyn_N,
                   'Pretrained': args.pretrained_model,
                   'Modelname': model_name,
                   'Patch_size': args.patch_size,
                   'Model_type': args.model_type
                   }

    # Test main function call
    fibseg_test_function.main(sect_name_dicts, eval_params, intermediate=args.intermediate,
                              model_dir=model_dir, load_case=args.cp_load_type, output_dir=out_dir,
                              verbose=args.verbose)

    if args.fiber_specs is True:
        fibseg_get_fiber_specs()



