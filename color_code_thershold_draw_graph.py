import ast
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import re
from plot_utils import *
import mpld3


def parse_tuple(value):
    return ast.literal_eval(value)

# Read the CSV, applying the converter to a specific column
df = pd.read_csv('data/threshold.csv', converters={'num_vortexes': parse_tuple})

# Display the first few rows of the DataFrame for verification
print(df.head())

# Choose which column to use for color and linestyle
color_column = 'dx'  # This can be 'd' or another column
# Create a color map for different code sizes
unique_d_list = sorted(df[color_column].unique())
color_map = cm.get_cmap('viridis', len(unique_d_list))
linestyle_column = 'num_vortexes'  # This can be 'vortex_location' or another column
marker_column = 'logical_operator_direction'  # This can be 'logical_operator_direction' or another column





# Create a new plot
plt.figure()

# # Filter the DataFrame to only include X logical operators and vortex_sign = '1'
df = df.query('logical_operator_direction == "x"')
# # draw pcolor of the logical error rate at the minimal physical error rate as a function of num_vortexes[0] and num_vortexes[1]
# df = df.query('phys_err_rate == phys_err_rate.min()')
#
# df[['vortex_x', 'vortex_y']] = pd.DataFrame(df['num_vortexes'].tolist(), index=df.index)
#
# # Loop through the logical_operator_direction values ('x' and 'y')
# for direction in df['logical_operator_direction'].unique():
#     # Filter the DataFrame for the current direction
#     df_filtered = df[df['logical_operator_direction'] == direction]
#
#     # Create a pivot table for plotting
#     # Index: vortex_x, Columns: vortex_y, Values: log_err_rate
#     pivot_table = df_filtered.pivot(index='vortex_x', columns='vortex_y', values='log_err_rate')
#
#     # Create the plot
#     plt.figure()
#     plt.pcolor(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='viridis')
#     plt.colorbar(label='log_err_rate')
#     plt.xlabel('vortex_y')
#     plt.ylabel('vortex_x')
#     plt.title(f'log_err_rate for direction: {direction}')
#
# plt.show()








# Group by the chosen columns and plot each group
for (color_val, num_vortexes, marker_val), group in df.groupby([color_column, linestyle_column, marker_column]):
    if num_vortexes == (0,0):
        linestyle = '-'
    elif num_vortexes == (1,0):
        linestyle = '--'
    elif num_vortexes == (0,1):
        linestyle = '-.'
    else:
        linestyle = ':'
    color = color_map(unique_d_list.index(color_val))  # Color based on code size 'd'
    marker = 'x' if marker_val == 'x' else 's'  # Marker for logical_operator_direction

    # Ensure data is sorted by phys_err_rate for proper line plotting
    group = group.sort_values(by='phys_err_rate')

    plt.errorbar(group['phys_err_rate'], group['log_err_rate'],
                 np.sqrt(group['log_err_rate'] * (1 - group['log_err_rate']) / group['shots']),
                 label=f'{color_val}, {num_vortexes}, {marker_val}',
                 linestyle=linestyle, marker=marker)#, color=color)  # Add marker for clarity

# Customize the plot
plt.xlabel('physical error rate')
plt.ylabel('logical error rate')
plt.yscale('log')
plt.xscale('log')
plt.legend(title=f'{color_column}, {linestyle_column}, {marker_column}', ncol=2)
edit_graph('Physical Error Rate', 'Logical Error Rate',
           scale=1.5)

# Show the plot
plt.savefig('figures/threshold.pdf')


plt.show()
