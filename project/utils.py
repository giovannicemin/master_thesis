import sys
import getopt


def get_params_from_cmdline(argv, default_params=None):
    '''Function that parse command line argments to dicitonary
    Parameters
    ----------
    argv : argv from sys.argv
    default_params : dict
        Dicionary to update

    Return
    ------
    Updated dictionary as in input
    '''
    arg_help = '{0} -L <length of spin chain> -b <beta> -V <potential> -w <working dir>'

    if default_params == None:
        raise Exception('Missing default parameters')

    try:
        opts, args = getopt.getopt(argv[1:], 'hL:b:V:w:', ['help', 'length', 'beta=', 'potential=', 'working_dir='])
    except:
        print(arg_help)
        sys.exit(2)

    for opts, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
        elif opt in ('-L', 'length'):
            default_params['L'] = arg
        elif opt in ('-b', 'beta'):
            default_params['beta'] = arg
        elif opt in ('-p', 'potential'):
            default_params['potential'] = arg
        elif opt in ('-w', 'working_dir'):
            default_params['working_dir'] = arg

    return default_params
