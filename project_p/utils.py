import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt


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

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
        elif opt in ('-L', '--length'):
            default_params['L'] = int(arg)
        elif opt in ('-b', '--beta'):
            default_params['beta'] = float(arg)
        elif opt in ('-p', '--potential'):
            default_params['potential'] = float(arg)
        elif opt in ('-w', '--working_dir'):
            default_params['working_dir'] = arg

    return default_params

def print_comparison(data_tebd, data_ml, T, dt, ylim):
    t = np.arange(0, T, dt)[:-1]

    rows= 5
    columns = 3

    names = ['I1X2', 'I1Y2', 'I1Z2', 'X1I2', 'X1X2', 'X1Y2', 'X1Z2', 'Y1I2', 'Y1X2', 'Y1Y2', 'Y1Z2', 'Z1I2', 'Z1X2', 'Z1Y2', 'Z1Z2']

    fig, axs = plt.subplots(rows, columns, figsize=(15,15), dpi=80)
    plt.setp(axs, xlim=(0,T), ylim=(-ylim, ylim))

    for i in range(rows):
        for j in range(columns):
            #if i == rows-1 & j == columns-1:
            #    continue
            axs[i, j].plot(t, [data_tebd[k][columns*i+j] for k in range(len(t))], label='Simulation', color='k')
            axs[i, j].plot(t, [data_ml[k][columns*i+j] for k in range(len(t))], label='ml', color='r', linestyle='--')
            axs[i, j].grid()
            axs[i, j].set_title(names[columns*i+j], x=0.5, y=0.85)
    axs[4, 2].grid()
    #plt.legend()
    plt.grid()

    fig = plt.gcf()
    plt.show()


def print_comparison_3(data_tebd, data_ml, data_ml_2, T, dt, ylim):
    t = np.arange(0, T, dt)[:-1]

    rows= 5
    columns = 3

    names = ['I1X2', 'I1Y2', 'I1Z2', 'X1I2', 'X1X2', 'X1Y2', 'X1Z2', 'Y1I2', 'Y1X2', 'Y1Y2', 'Y1Z2', 'Z1I2', 'Z1X2', 'Z1Y2', 'Z1Z2']

    fig, axs = plt.subplots(rows, columns, figsize=(15,15), dpi=80)
    plt.setp(axs, xlim=(0,T), ylim=(-ylim, ylim))

    for i in range(rows):
        for j in range(columns):
            #if i == rows-1 & j == columns-1:
            #    continue
            axs[i, j].plot(t, [data_tebd[k][columns*i+j] for k in range(len(t))], label='Simulation', color='k')
            axs[i, j].plot(t, [data_ml_2[k][columns*i+j] for k in range(len(t))], label='ml', color='b', linestyle='-', alpha=0.6)
            axs[i, j].plot(t, [data_ml[k][columns*i+j] for k in range(len(t))], label='ml', color='r', linestyle='--')
            axs[i, j].grid()
            axs[i, j].set_title(names[columns*i+j], x=0.5, y=0.85)
    axs[4, 2].grid()
    #plt.legend()
    plt.grid()

    fig = plt.gcf()
    plt.show()


def print_omega(model, T, ylim=1, dt=0.01):
    time = np.arange(0, T, dt)

    rows= 5
    columns = 3

    fig, axs = plt.subplots(rows, columns, figsize=(15,15), dpi=80)
    plt.setp(axs, xlim=(0,T), ylim=(-ylim, ylim))

    model.eval()
    for i in range(rows):
        for j in range(columns):
            omega_learned = np.array([model.MLP.get_omega(t)[columns*i+j] for t in time])
            axs[i, j].plot(time, omega_learned, label='ml', color='r', linestyle='--')
            axs[i, j].grid()
            #axs[i, j].set_title(names[columns*i+j], x=0.5, y=0.85)
    axs[4, 2].grid()
    #plt.legend()
    plt.grid()

    fig = plt.gcf()
    plt.show()


def print_rates(model, T, ylim=1, dt=0.01):
    time = np.arange(0, T, dt)

    rows= 5
    columns = 3

    fig, axs = plt.subplots(rows, columns, figsize=(15,15), dpi=80)
    plt.setp(axs, xlim=(0,T), ylim=(-ylim, ylim))

    model.eval()
    for i in range(rows):
        for j in range(columns):
            omega_learned = np.array([model.MLP.get_rates(t)[columns*i+j] for t in time])
            axs[i, j].plot(time, omega_learned, label='ml', color='r', linestyle='--')
            axs[i, j].grid()
            #axs[i, j].set_title(names[columns*i+j], x=0.5, y=0.85)
    axs[4, 2].grid()
    #plt.legend()
    plt.grid()

    fig = plt.gcf()
    plt.show()
