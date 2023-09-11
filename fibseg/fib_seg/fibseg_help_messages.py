import pkg_resources

#=========================================================================================
# Fibseg help, description and epilog messages to display
# Vaanathi Sundaresan
# 09-03-2023, Oxford
#=========================================================================================


def help_descs():
    version = pkg_resources.require("fibseg")[0].version
    helps = {
        'mainparser':
        "fibseg: Triplanar ensemble U-Net model, v" + str(version) + "\n" 
        "   \n" 
        "Sub-commands available:\n" 
        "       fibseg apply           Applying a saved/pretrained TrUE-Net model for testing\n"
        "   \n"
        "   \n"
        "For detailed help regarding the options for each command,\n"
        "type fibseg <command> --help (e.g. fibseg evaluate --help)\n"
        "   \n",

        'evaluate':
        'fibseg evaluate: testing the TrUE-Net model, v' + str(version) + '\n'
        '   \n'
        'Usage: fibseg evaluate -i <input_masterfile> -m <model_name> -o <output_directory> [options]'
        '   \n'
        'Compulsory arguments:\n'
        '       -i, --inp_file                        Name of the masterfile with the absolute path\n'
        '       -m, --model_name                      Model basename with absolute path / '
        '                                                   pretrained model name (mwsc, mwsc_flair, mwsc_t1, ukbb, ukbb_flair, ukbb_t1)\n'                                                                  
        '       -o, --output_dir                      Path to the directory for saving output predictions\n'
        '   \n'
        'Optional arguments:\n'
        '       -int, --intermediate                  Saving intermediate prediction results (individual planes) for each subject [default = False]\n'
        '       -cpu, --use_cpu                       Force fibseg to run on CPU (default=False)' 
        '       -v, --verbose                         Display debug messages [default = False]\n'
        '   \n'
    }
    return helps


def desc_descs():
    version = pkg_resources.require("fibseg")[0].version
    descs = {
        'mainparser' :
        "fibseg: Fiber bundle segmentation model, v" + str(version) + "\n"
        "   \n"
        "Sub-commands available:\n"
        "       fibseg evaluate        Applying a saved/pretrained TrUE-Net model for testing\n"
        "   \n",

        'evaluate':
        '   \n'
        'fibseg: Fiber bundle segmentation, v' + str(version) + '\n'
        '   \n'
    }
    return descs


def epilog_descs():
    epilogs = {
        'mainparser' :
        "   \n"
        "For detailed help regarding the options for each command,\n"
        "type fibseg <command> --help or -h (e.g. fibseg train --help, fibseg train -h)\n"
        "   \n",

        'subparsers' :
        '   \n'
        "For detailed help regarding the options for each argument,\n"
        "refer to the user-guide or readme document. For more details on\n"
        "FibSeg, refer to https://doi.org/10.1007/978-3-031-16961-8_12\n"
        "   \n",
    }
    return epilogs

