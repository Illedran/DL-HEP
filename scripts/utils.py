import numpy as np
import matplotlib.pyplot as plt

def batch_generator(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def ams(y_true, y_predict, weights):
    s = weights[np.logical_and(y_true, y_predict)].sum()
    b = weights[np.logical_and(np.logical_not(y_true), y_predict)].sum()
    b_reg = 10

    return np.sqrt(2 * ((s + b + b_reg) * np.log(s / (b + b_reg) + 1) - s))

def show_plots(name, save_to='', format='pdf', dpi=300):
    import os
    if 'DISPLAY' in os.environ:
        DISPLAY = True
    else:
        DISPLAY = False

    if DISPLAY and save_to == '':
        plt.show()
    else:
        plt.savefig(save_to + '/' + name + '.{}'.format(format), format=format, dpi=dpi)
