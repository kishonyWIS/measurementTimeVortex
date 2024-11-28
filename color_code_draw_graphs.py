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




df = df.drop(columns=['detectors', 'logical_operator_pauli_type', 'reps_with_noise'])

df_x = df[df['logical_operator_direction'] == 'x'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_x'})
df_y = df[df['logical_operator_direction'] == 'y'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_y'})
df_both = df[df['logical_operator_direction'] == 'both'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_both'})

# Merge the two DataFrames on all columns except for 'log_err_rate_x' and 'log_err_rate_y'
merge_columns = [col for col in df_x.columns if col not in ['log_err_rate_x', 'log_err_rate_y']]

# Perform the merge
df = pd.merge(df_x, df_y, on=merge_columns)
df = pd.merge(df, df_both, on=merge_columns)




df['N'] = df['dx'] * df['dy'] * 4
# Choose which column to use for color and linestyle
color_column = 'N'  # This can be 'd' or another column
# Create a color map for different code sizes
unique_d_list = sorted(df[color_column].unique())
color_map = cm.get_cmap('viridis', len(unique_d_list))
linestyle_column = 'num_vortexes'  # This can be 'vortex_location' or another column
marker_column = 'logical_operator_direction'  # This can be 'logical_operator_direction' or another column

# # Filter the DataFrame to only include X logical operators and vortex_sign = '1'



for logical_operator_direction in ['x', 'y', 'both']:

    # Create a new plot
    plt.figure(figsize=(15,10))

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






# filter by physical error rate == 0.005
# plot for each N the minimal logical error rate for "both" logical operator direction labeled by the number of vortexes and dx, dy
# also plot for each N the minimal logical error rate for "both" with number of vortexes = (0,0) and label by dx, dy

for logical_operator_direction in ['x', 'y', 'both']:
    df_temp = df.query('phys_err_rate == 0.005')
    df_temp = df_temp.drop(columns='phys_err_rate')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for N, group in df_temp.groupby('N'):
        # check if N is a square number
        # if int(np.sqrt(N/24))**2 != N/24:
        #     continue
        minimal_row = group.loc[group['log_err_rate_both'].idxmin()]
        ax.loglog(minimal_row['N'], minimal_row['log_err_rate_'+logical_operator_direction], 'o', label=f'dx={minimal_row["dx"]}, dy={minimal_row["dy"]}, nv={minimal_row["num_vortexes"]}')
        minimal_row = group.loc[group['num_vortexes'] == (0,0)]
        minimal_row = minimal_row.loc[minimal_row['log_err_rate_both'].idxmin()]
        ax.loglog(minimal_row['N'], minimal_row['log_err_rate_'+logical_operator_direction], 'x', label=f'dx={minimal_row["dx"]}, dy={minimal_row["dy"]}, nv={minimal_row["num_vortexes"]}')
    # put the legend outside the plot
    plt.legend(title='dx, dy, nv', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('N')
    plt.ylabel('Minimal Logical Error Rate')
    plt.title(f'Logical Operator Direction: {logical_operator_direction}')
    plt.tight_layout()
plt.show()



# plt.show()
    # Show the plot
# plt.savefig('figures/threshold.pdf')

# plot a pcolor graph of the minimal logical error rate for each vortex configuration

# split the vortex configurations into two columns
df['num_vortexes_x'] = df['num_vortexes'].apply(lambda x: x[0])
df['num_vortexes_y'] = df['num_vortexes'].apply(lambda x: x[1])
df = df.drop(columns='num_vortexes')

# take the maximum system size
# df_temp = df.query(f'dx == {max(df.dx)} and dy == {max(df.dy)}')
# df_temp = df.query(f'dx == 4 and dy == 6')

# iterate over all combinations of dx and dy
for dx, dy in df[['dx', 'dy']].drop_duplicates().values:
    print(f'dx={dx}, dy={dy}')
    df_temp = df.query(f'dx == {dx} and dy == {dy}')

    fig, axes = plt.subplots(3, 1, figsize=(5, 12))

# pivot the DataFrame to have the vortex configurations as the index
    for logical_operator_direction, ax in zip(['x', 'y', 'both'], axes):

        df_temp_direction = df_temp.pivot(index=['phys_err_rate'], columns=['num_vortexes_x', 'num_vortexes_y'], values='log_err_rate_'+logical_operator_direction)
        df_temp_direction = df_temp_direction.min()
        df_temp_direction = df_temp_direction.unstack()
        x, y = df_temp_direction.index, df_temp_direction.columns
        z = df_temp_direction.values
        # draw a 2d graph of the minimal logical error rate for each vortex configuration

        # set current axis
        im = ax.pcolor(x, y, z.T, cmap='viridis')
        # plt.pcolor(x, y, z.T, cmap='viridis')
        plt.xlabel('Number of X Vortexes')
        ax.set_ylabel('Number of Y Vortexes')
        # set the ticks to the values of the vortexes
        ax.set_xticks(np.unique(x))
        ax.set_yticks(np.unique(y))
        plt.colorbar(im, ax=ax)
        # move the title up a bit
        ax.set_title(f'dx={dx}, dy={dy}, logical_operator_direction={logical_operator_direction}', y=1.1)
    plt.tight_layout()
plt.show()
