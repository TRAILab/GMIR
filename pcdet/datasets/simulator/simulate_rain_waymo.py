
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_colors(pc, color_feature=None):
    # create colormap
    if color_feature == 0:
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 1:

        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 2:
        feature = pc[:, 2]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 3:
        feature = pc[:, 3]
        min_value = 0
        max_value = 255

    elif color_feature == 4:
        feature = pc[:, 4]
        colors = np.zeros((feature.shape[0], 3))

        colors[feature == 0, 0] = 1 #lost red
        colors[feature == 1, 1] = 1 # scattered green
        colors[feature == 2, 2] = 1 # original but noisy blue

        #colors[feature == 3, 1] = 1 # random scatter
        print(f'\nlost % #: {100 * np.count_nonzero(colors[:,0])/ pc.shape[0]}')
        print(f'scattered % #: {100 * np.count_nonzero(colors[:, 1])/ pc.shape[0]}')
        print(f'attenuated % #: {100 * np.count_nonzero(colors[:, 2])/ pc.shape[0]}')


        # min_value = np.min(feature)
        # max_value = np.max(feature)

    else:
        feature = np.linalg.norm(pc[:, 0:3], axis=1)
        min_value = np.min(feature)
        max_value = np.max(feature)



    # norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    #
    #
    # cmap = cm.jet  # sequential
    #
    # m = cm.ScalarMappable(norm=norm, cmap=cmap)
    #
    # colors = m.to_rgba(feature)
    # colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    # colors[:, 3] = 0.5



    return colors[:, :3]

def genHistogram(pc, Rr=0):
    plt.close('all')
    fig = plt.figure('hist', figsize=(10,10))
    range = np.linalg.norm(pc[:, 0:3], axis=1)
    intensity = pc[:,3]

    fig.add_subplot(1,2,1)
    n, bins, patches = plt.hist(x=range,
                                bins=np.linspace(start=0.1, stop=min(range.max(), 100), num=20 + 1, endpoint=True),
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Range')
    plt.ylabel('Number of Points')
    plt.title(f'Visibility of Point Cloud at Rr {Rr}')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


    fig.add_subplot(1, 2, 2)
    n, bins, patches = plt.hist(x=intensity,
                                bins=np.linspace(start=0, stop=max(intensity.max(), 1), num=20 + 1, endpoint=True),
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)


    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Number of Points')
    plt.title(f'Intensities of Point Cloud')
    plt.savefig('hist_Rr.png')
    plt.show()
