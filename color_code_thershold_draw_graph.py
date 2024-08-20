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
df = df.drop(columns=['Unnamed: 0'])

# Display the first few rows of the DataFrame for verification
print(df.head())






df_x = df[df['logical_operator_direction'] == 'x'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_x'})
df_y = df[df['logical_operator_direction'] == 'y'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_y'})

# Merge the two DataFrames on all columns except for 'log_err_rate_x' and 'log_err_rate_y'
merge_columns = [col for col in df_x.columns if col not in ['log_err_rate_x', 'log_err_rate_y']]

# Perform the merge
df = pd.merge(df_x, df_y, on=merge_columns)
df['log_err_rate_both'] = 1 - (1 - df['log_err_rate_x']) * (1 - df['log_err_rate_y'])






# Choose which column to use for color and linestyle
color_column = 'dx'  # This can be 'd' or another column
# Create a color map for different code sizes
unique_d_list = sorted(df[color_column].unique())
color_map = cm.get_cmap('viridis', len(unique_d_list))
linestyle_column = 'num_vortexes'  # This can be 'vortex_location' or another column
marker_column = 'logical_operator_direction'  # This can be 'logical_operator_direction' or another column

# # Filter the DataFrame to only include X logical operators and vortex_sign = '1'
df = df.query('geometry == "SymmetricTorus"')


for logical_operator_direction in ['x', 'y', 'both']:

    # Create a new plot
    plt.figure()

    # Group by the chosen columns and plot each group
    for (color_val, num_vortexes), group in df.groupby([color_column, linestyle_column]):
        if num_vortexes == (0,0):
            linestyle = '-'
        elif num_vortexes == (1,0):
            linestyle = '--'
        elif num_vortexes == (0,1):
            linestyle = '-.'
        else:
            linestyle = ':'

        group = group.rename(columns={'log_err_rate_'+logical_operator_direction: 'log_err_rate'})

        color = color_map(unique_d_list.index(color_val))  # Color based on code size 'd'
        if logical_operator_direction == 'x':
            marker = 'x'
        elif logical_operator_direction == 'y':
            marker = 'o'
        elif logical_operator_direction == 'both':
            marker = 's'

        # Ensure data is sorted by phys_err_rate for proper line plotting
        group = group.sort_values(by='phys_err_rate')

        plt.errorbar(group['phys_err_rate'], group['log_err_rate'],
                     np.sqrt(group['log_err_rate'] * (1 - group['log_err_rate']) / group['shots']),
                     label=f'{color_val}, {num_vortexes}, {logical_operator_direction}',
                     linestyle=linestyle, marker=marker, color=color)  # Add marker for clarity

    # Customize the plot
    plt.xlabel('physical error rate')
    plt.ylabel('logical error rate')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(title=f'{color_column}, {linestyle_column}, {marker_column}', ncol=3)
    edit_graph('Physical Error Rate', 'Logical Error Rate',
               scale=1.5)

    # Show the plot
# plt.savefig('figures/threshold.pdf')

# plot a pcolor graph of the minimal logical error rate for each vortex configuration

# split the vortex configurations into two columns
df['num_vortexes_x'] = df['num_vortexes'].apply(lambda x: x[0])
df['num_vortexes_y'] = df['num_vortexes'].apply(lambda x: x[1])
df = df.drop(columns='num_vortexes')

df_temp = df.query('dx == 9 and dy == 9')

# pivot the DataFrame to have the vortex configurations as the index
for logical_operator_direction in ['x', 'y', 'both']:

    df = df_temp.pivot(index=['phys_err_rate'], columns=['num_vortexes_x', 'num_vortexes_y'], values='log_err_rate_'+logical_operator_direction)
    df = df.min()
    df = df.unstack()
    x, y = np.meshgrid(df.columns, df.index)
    z = df.values
    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, z, cmap='viridis')
    plt.colorbar(c, label='logical error rate'+' '+logical_operator_direction)
    plt.xlabel('Number of Y Vortexes')
    plt.ylabel('Number of X Vortexes')
plt.show()
