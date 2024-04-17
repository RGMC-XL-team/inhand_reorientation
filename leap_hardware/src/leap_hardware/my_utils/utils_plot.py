from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import numpy as np

# -------------------------------------------------------
# 用于绘制axis比例相同的三维图
def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])



# -------------------------------------------------------
"""
    Input:
        data: shape (m, n)
"""
def plotHistogram(data, x_labels, bar_labels, bar_colors, 
                  data_bottom=None,
                  x_width=0.8, bar_interval_ratio=0.0, border_width=0.0, 
                  edgecolor='k', linewidth=1, 
                  use_default_setting=True):

    data = np.array(data)
    if data.shape[0] != len(x_labels) or data.shape[1] != len(bar_labels):
        print("The dimension of input data is wrong.")
        return False
    
    if data_bottom is None:
        data_bottom = np.zeros(data.shape)
    elif data_bottom.shape != data.shape:
        print("The dimension of input data_bottom is wrong.")
        return False

    x = np.arange(len(x_labels))
    m = len(x_labels)
    n = len(bar_labels)
    if n == 1: bar_interval = 0.0
    else: bar_interval = (x_width * bar_interval_ratio) / float(n-1)
    bar_width = (x_width - (n-1) * bar_interval) / n

    plt.bar(x + (n-1)/2.0 * (bar_width + bar_interval), np.zeros((data.shape[0],)), width=bar_width, tick_label=x_labels)
    for i in range(n):
        plt.bar(x + i * (bar_width + bar_interval), data[:, i], bottom=data_bottom[:, i], width=bar_width, label=bar_labels[i], 
                color=bar_colors[i], edgecolor=edgecolor, linewidth=linewidth)

    if use_default_setting:
        # plt.xticks([])
        # plt.grid()
        plt.xlim([0 - bar_width/2.0 - border_width , (m-1) + (n-1) * (bar_width + bar_interval) + bar_width/2.0 + border_width])
        

# ----------------------------------------------------------
def changeTickerDensity(xaxis=None, yaxis=None):
    ax = plt.gca()
    if xaxis is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xaxis))
    if yaxis is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yaxis))


# # ----------------------------------------------------------
# def plotGrid():
