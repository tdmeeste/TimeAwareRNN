import numpy as np
import os
import itertools
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        #print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w') as fID:
            fID.write('%s\n'%header)
        self.f = f

    def __call__(self, *args):
        #standard output
        print(*args)
        #log to file
        try:
            with open(self.f, 'a') as fID:
                fID.write(' '.join(str(a) for a in args)+'\n')
        except:
            print('Warning: could not log to', self.f)


def show_data(t, target, pred, folder, tag, msg=''):

    plt.figure(1)
    maxv = np.max(target)
    minv = np.min(target)
    view = maxv - minv

    # linear
    n = target.shape[1]
    for i in range(n):
        ax_i = plt.subplot(n, 1, i+1)
        plt.plot(t, target[:, i], 'g--')
        plt.plot(t, pred[:, i], 'r.')
        #ax_i.set_ylim(minv - view/10, maxv + view/10)
        if i == 0:
            plt.title(msg)

    #fig, axs = plt.subplots(6, 1)
    #for i, ax in enumerate(axs):
    #    ax.plot(target[:, i], 'g--', pred[:, i], 'r-')

    plt.savefig("%s/%s.png"%(folder, tag))
    plt.close('all')