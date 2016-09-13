import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import subprocess
import sys
import tempfile


def get_fig_size(fig_w_cm, fig_h_cm=None):
    if not fig_h_cm:
        golden_ratio = (1 + math.sqrt(5))/2
        fig_h_cm = fig_w_cm / golden_ratio

    size_cm = (fig_w_cm, fig_h_cm)
    return map(lambda x: x/2.54, size_cm)


def label_size():
    return 8


def font_size():
    return 8


def ticks_size():
    return 7


def axis_lw():
    return 0.6


def plot_lw():
    return 1.5


def axis_color():
    return 'k'  # [0.5]*3


def grid_color():
    return [0.8]*3


def grid_lw():
    return 0.2


def color(key=None):
    colors = {'g': np.array([77, 175, 74])/255.0,
              'b': np.array([55, 126, 184])/255.0,
              'db': np.array([8, 48, 108])/255.0,
              'lb': np.array([182, 211, 232])/255.0,
              'r': np.array([228, 26, 28])/255.0,
              'p': np.array([152, 78, 163])/255.0,
              'o': np.array([255, 127, 0])/255.0,
              'br': np.array([166, 86, 40])/255.0,
              'pk': np.array([247, 129, 191])/255.0}
    if not key:
        return colors
    else:
        return colors[key]


def figure_setup():
    # use latex-style fonts for plots
    params = {'text.usetex': True,
              'figure.dpi': 200,
              'font.size': font_size(),
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'axes.labelsize': label_size(),
              'axes.labelcolor': axis_color(),
              'axes.titlesize': font_size(),
              'axes.linewidth': axis_lw(),
              'axes.edgecolor': axis_color(),
              'text.fontsize': font_size(),
              'legend.fontsize': font_size(),
              'xtick.color': axis_color(),
              'xtick.direction': 'out',
              'xtick.labelsize': ticks_size(),
              'ytick.color': axis_color(),
              'ytick.direction': 'out',
              'ytick.labelsize': ticks_size(),
              'font.family': 'serif'}
    plt.rcParams.update(params)


def save_fig(fig, file_name, fmt='pdf', dpi=300, tight=True):
    """Saves a Matplotlib figure as EPS/PNG/PDF to the given path and trims it.
    """

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    print 'Saving figure to %s...' % (color_print(file_name, 'light-gray'))

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)


def get_rmse(f):
    data_dir = '../results'
    data = sio.loadmat('%s/%s' % (data_dir, f))
    obs = data['obs'][0]
    mse = data['model_test_mse']

    n_obs = len(obs)
    n_votes = mse.shape[1]
    n_runs = mse.shape[2]
    avg_rmse = np.mean(np.sqrt(np.reshape(mse, (n_obs, n_votes * n_runs))), axis=1)

    return obs, avg_rmse*100


def get_mae(f):
    data_dir = '../results'
    data = sio.loadmat('%s/%s' % (data_dir, f))
    obs = data['obs'][0]
    mae = data['model_test_mae'][0]

    n_obs = len(obs)
    n_votes = mse.shape[1]
    n_runs = mse.shape[2]

    avg_mae = np.mean(np.reshape(mae, (n_obs, n_votes * n_runs)), axis=1)

    return obs, avg_mae * 100


def get_accuracy(f):
    data_dir = '../results'
    data = sio.loadmat('%s/%s' % (data_dir, f))
    obs = data['obs'][0]
    accuracy = data['model_test_mae'][1]

    n_obs = len(obs)
    n_votes = mse.shape[1]
    n_runs = mse.shape[2]

    avg_acc = np.mean(np.reshape(accuracy, (n_obs, n_votes * n_runs)), axis=1)

    return obs, avg_acc * 100


def plot_models_results(models, order, get_data, save, ncol=1, national=False,
                        accuracy=False):
    options = {
        ':': {'dashes': (2.5, 2.5)},
        '-.': {'dashes': (6, 3, 2, 3)},
        '--': {'dashes': (7, 5)},
        '-': {},
    }

    figure_setup()

    figsize = get_fig_size(8.7, 5.2)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for k in order:
        ls, f = models[k]
        x, y = get_data(f)
        ax.semilogx(x, y, label=k, ls=ls, c='k', lw=plot_lw(), **options[ls])

    ax.set_axisbelow(True)

    ax.set_xlim([1, 2116])
    if not accuracy:
        if national:
            ax.set_ylim([0, 8])
        else:
            ax.set_ylim([4.5, 13])

    ax.set_xlabel('Number of observed regions')
    if accuracy:
        ax.set_ylabel('Accuracy of national result [\\%]')
    elif national:
        ax.set_ylabel('RMSE on national result [\\%]')
    else:
        ax.set_ylabel('RMSE on test regions [\\%]')

    loc = 'upper right'
    if accuracy:
        loc = 'lower right'

    leg = plt.legend(fontsize=label_size(), loc=loc, ncol=ncol,
                     handlelength=3, handletextpad=0.1, columnspacing=1,
                     labelspacing=0.2)
    leg.get_frame().set_linewidth(0)

    plt.grid(True, which='major', c=grid_color(), ls='-', lw=grid_lw())
    plt.grid(True, which='minor', c=grid_color(), ls='-', lw=grid_lw())

    plt.tight_layout()

    if save:
        save_fig(fig, save, 'eps')
    else:
        plt.show()


colorMapping = {
    'black': '0;30',
    'dark-gray': '1;30',
    'blue': '0;34',
    'light-blue': '1;34',
    'green': '0;32',
    'light-green': '1;32',
    'cyan': '0;36',
    'light-cyan': '1;36',
    'red': '0;31',
    'light-red': '1;31',
    'purple': '0;35',
    'light-purple': '1;35',
    'brown': '0;33',
    'yellow': '1;33',
    'light-gray': '0;37',
    'white': '1;37',
}


def progress(count, total):
    if count > 0:
        sys.stdout.write('\b' * 7)

    sys.stdout.write(('%.2f%%' % (count * 100.0 / (total - 1),)).rjust(7))

    if count == total - 1:
        sys.stdout.write('\n')

    sys.stdout.flush()


def color_print(txt, color, fmt='%s'):
    if color is None:
        return fmt % (txt,)
    else:
        return ('\033[%sm' + fmt + '\033[0m') % (colorMapping[color], txt)
