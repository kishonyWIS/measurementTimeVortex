import ast
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import title
from matplotlib.lines import Line2D

from plot_utils import *

def parse_tuple(value):
    return ast.literal_eval(value)

# Read the CSV, applying the converter to a specific column
df = pd.read_csv('data/data_threshold_different_vortices.csv', converters={'num_vortexes': parse_tuple})
df = df.drop(columns=['Unnamed: 0'])

# draw the logical error rate "both" vs the system size for each error rate. Use full lines for no vortexes and dashed lines for 2L-1 vortexes.

df = df.drop(columns=['detectors', 'logical_operator_pauli_type', 'reps_with_noise'])

# filter a single system size
L = 4
df = df.query(f'dx == {2*L} and dy == {3*L}')
# for each vortex configuration, draw the logical error rate as a function of the physical error rate
# Create a new plot
for logical_operator_direction in ['x', 'y', 'both']:
    plt.figure()
    # reset colors
    plt.gca().set_prop_cycle(None)
    color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # reset markers cycle
    plt.gca().set_prop_cycle(None)
    marker_cycle = list(Line2D.filled_markers)
    vx_list = list(np.arange(min(df['num_vortexes'].apply(lambda x: x[0])), max(df['num_vortexes'].apply(lambda x: x[0])) + 1))
    vy_list = list(np.arange(min(df['num_vortexes'].apply(lambda x: x[1])), max(df['num_vortexes'].apply(lambda x: x[1])) + 1))
    slopes = np.nan * np.zeros((len(vx_list), len(vy_list)))
    interceps = np.nan * np.zeros((len(vx_list), len(vy_list)))
    # Group by the chosen columns and plot each group
    for name, group in df[df['logical_operator_direction'] == logical_operator_direction].groupby('num_vortexes'):
        color = color_cycle[vx_list.index(name[0])]
        marker = marker_cycle[vy_list.index(name[1])]
        plt.loglog(group['phys_err_rate'], group['log_err_rate'], label=f'{name}', linestyle='-', marker=marker, color=color)
        # calculate a linear fit with the values with log_err_rate between 1e-5 and 1e-1
        x = group['phys_err_rate'][group['log_err_rate'] > 1e-4][group['log_err_rate'] < 1e-1]
        y = group['log_err_rate'][group['log_err_rate'] > 1e-4][group['log_err_rate'] < 1e-1]
        m, b = np.polyfit(np.log(x), np.log(y), 1)
        slopes[vx_list.index(name[0]), vy_list.index(name[1])] = m
        interceps[vx_list.index(name[0]), vy_list.index(name[1])] = b
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.legend()
    plt.title(f'Logical error rate for {logical_operator_direction} logical operator')

    # create a pcolor of the slopes for each vortex configuration
    plt.figure()
    plt.pcolor(vx_list, vy_list, slopes.T, shading='auto', cmap='viridis')
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.title(f'Slopes for {logical_operator_direction} logical operator')
    plt.colorbar()

    # create a pcolor for the logical error rate extrapolated to physical error rate = 1e-4
    plt.figure()
    # calculate the logical error rate at physical error rate 1e-4 in log scale
    log_err_rate_1e4 = np.log10(np.exp(slopes * np.log(1e-4) + interceps))
    plt.pcolor(vx_list, vy_list, log_err_rate_1e4.T, shading='auto', cmap='viridis')
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.title(f'Logical error rate at physical error rate 1e-4 for {logical_operator_direction} logical operator')
    plt.colorbar()

plt.show()