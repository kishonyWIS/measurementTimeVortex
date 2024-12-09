import ast
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import title
from matplotlib.lines import Line2D

from plot_utils import *

def parse_tuple(value):
    return ast.literal_eval(value)

# Read the CSV, applying the converter to a specific column
df = pd.read_csv('data/data_threshold.csv', converters={'num_vortexes': parse_tuple})
df = df.drop(columns=['Unnamed: 0'])

# draw the logical error rate "both" vs the system size for each error rate. Use full lines for no vortexes and dashed lines for 2L-1 vortexes.

df = df.drop(columns=['detectors', 'logical_operator_pauli_type', 'reps_with_noise'])

df_without_vortexes = df[df['num_vortexes'] == (0,0)]
df_with_vortexes = df[df['num_vortexes'] != (0, 0)]

for df in [df_without_vortexes, df_with_vortexes]:
    # on the same graph, draw the x, y and both logical error rates as a function of the physical error rate for each system size
    # Create a new plot
    plt.figure()
    # reset colors
    plt.gca().set_prop_cycle(None)
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Group by the chosen columns and plot each group
    for name, group in df.groupby('dx'):
        color = next(color_cycle)
        for logical_operator_direction, linestyle in zip(['x', 'y', 'both'], ['-', '--', ':']):
            group_logical = group[group['logical_operator_direction'] == logical_operator_direction]
            plt.loglog(group_logical['phys_err_rate'], group_logical['log_err_rate'], label=f'{logical_operator_direction} L={name}', linestyle=linestyle, color=color)
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.legend()

    # draw the ratio between the x and y logical error rates as a function of the physical error rate for each system size
    # Create a new plot
    plt.figure()
    # reset colors
    plt.gca().set_prop_cycle(None)
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Group by the chosen columns and plot each group
    for name, group in df.groupby('dx'):
        color = next(color_cycle)
        group_x = group[group['logical_operator_direction'] == 'x']
        group_y = group[group['logical_operator_direction'] == 'y']
        plt.loglog(group_x['phys_err_rate'], group_x['log_err_rate'].values / group_y['log_err_rate'].values, label=f'L={name}', color=color)
    plt.xlabel('Physical error rate')
    plt.ylabel('Ratio between x and y logical error rates')
    plt.legend()

plt.show()