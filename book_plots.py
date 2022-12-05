# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:21:40 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_errorbars(bars, xlims):

    i = 1.0
    for bar in bars:
        plt.errorbar([bar[0]], [i], xerr=[bar[1]], fmt='o',
                     label=bar[2], capthick=2, capsize=10)
        i += 0.2

    plt.ylim(0, 2)
    plt.xlim(xlims[0], xlims[1])
    show_legend()
    plt.gca().axes.yaxis.set_ticks([])
    plt.show()


def show_legend():
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def bar_plot(pos, ylim=(0, 1), title=None):
    plt.cla()
    ax = plt.gca()
    x = np.arange(len(pos))
    ax.bar(x, pos, color='#30a2da')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x+0.4, x)
    if title is not None:
        plt.title(title)


def set_labels(title=None, x=None, y=None):
    """ helps make code in book shorter. Optional set title, xlabel and ylabel
    """
    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)


def set_limits(x, y):
    """ helper function to make code in book shorter. Set the limits for the x
    and y axis.
    """

    plt.gca().set_xlim(x)
    plt.gca().set_ylim(y)


def plot_measurements(xs, ys=None, color='r', lw=2, label='Measurements', **kwargs):
    """ Helper function to give a consistant way to display
    measurements in the book.
    """

    plt.autoscale(tight=True)
    '''if ys is not None:
        plt.scatter(xs, ys, marker=marker, c=c, s=s,
                    label=label, alpha=alpha)
        if connect:
           plt.plot(xs, ys, c=c, lw=1, alpha=alpha)
    else:
        plt.scatter(range(len(xs)), xs, marker=marker, c=c, s=s,
                    label=label, alpha=alpha)
        if connect:
           plt.plot(range(len(xs)), xs, lw=1, c=c, alpha=alpha)'''

    if ys is not None:
        plt.plot(xs, ys, color=color, lw=lw, ls='--', label=label, **kwargs)
    else:
        plt.plot(xs, color=color, lw=lw, ls='--', label=label, **kwargs)


def plot_residual_limits(Ps, stds=1.):
    """ plots standand deviation given in Ps as a yellow shaded region. One std
    by default, use stds for a different choice (e.g. stds=3 for 3 standard
    deviations.
    """

    std = np.sqrt(Ps) * stds

    plt.plot(-std, color='k', ls=':', lw=2)
    plt.plot(std, color='k', ls=':', lw=2)
    plt.fill_between(range(len(std)), -std, std,
                     facecolor='#ffff00', alpha=0.3)

    return std


def plot_track(xs, ys=None, label='Track', c='k', lw=2, **kwargs):
    if ys is not None:
        plt.plot(xs, ys, color=c, lw=lw, ls=':', label=label, **kwargs)
    else:
        plt.plot(xs, color=c, lw=lw, ls=':', label=label, **kwargs)


def plot_filter(xs, ys=None, c='#013afe', label='Filter', vars=None, **kwargs):
    # def plot_filter(xs, ys=None, c='#6d904f', label='Filter', vars=None, **kwargs):

    if ys is None:
        ys = xs
        xs = range(len(ys))

    plt.plot(xs, ys, color=c, label=label, **kwargs)

    if vars is None:
        return
    vars = np.asarray(vars)

    std = np.sqrt(vars)
    std_top = ys+std
    std_btm = ys-std

    plt.plot(xs, ys+std, linestyle=':', color='k', lw=2)
    plt.plot(xs, ys-std, linestyle=':', color='k', lw=2)
    plt.fill_between(xs, std_btm, std_top,
                     facecolor='yellow', alpha=0.2)


def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(W, maxweight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()

    plt.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')

    plt.axis('off')
    plt.axis('equal')
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, -w/maxweight),
                      'black')
    if reenable:
        plt.ion()


def plot_residuals(xs, filter_xs, Ps, title, y_label, stds=1):
    res = xs - filter_xs
    plt.plot(res, 'g+')
    ps_std = plot_residual_limits(Ps, stds)
    set_labels(title, 'time (sec)', y_label)
    plt.show()

    # density
    positive_delta = ps_std-abs(res)
    positive_delta = positive_delta[positive_delta >= 0]
    total = res.size
    one_sigma_counter = positive_delta.size
    # if abs
    print(
        f'% valid predicted values: {np.round_(one_sigma_counter/total*100,decimals=2)}%')
    if one_sigma_counter/total >= 0.68:
        print(f'>good fit filter model: std = {np.mean(ps_std)}')
    else:
        print(f'>bad fit filter model: std = {np.mean(ps_std)}')

# if __name__ == "__main__":
#     p = [0.2245871, 0.06288015, 0.06109133, 0.0581008, 0.09334062, 0.2245871,
#      0.06288015, 0.06109133, 0.0581008,  0.09334062]*2
#     bar_plot(p)
#     plot_measurements(p)
