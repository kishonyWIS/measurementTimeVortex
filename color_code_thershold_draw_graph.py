import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import re
from plot_utils import *
import mpld3

# Directory where the pickle files are saved
data_dir = 'data'

# Regular expression pattern to extract parameters from filenames
pattern = re.compile(
    r'threshold_noisetype_(?P<noise_type>[^_]+)_'
    r'logical_operator_(?P<logical_operator_pauli_type>[^_]+)_'
    r'direction_(?P<logical_operator_direction>[^_]+)_'
    r'boundary_conditions_(?P<boundary_conditions>[^_]+)_'
    r'vortex_(?P<vortex_location>[^_]+)_(?P<vortex_sign>[^_]+)\.pkl'
)

# Initialize an empty list to collect rows of data for the DataFrame
rows = []

# Loop through all the pickle files in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Extract parameters from the filename using regex
        match = pattern.match(filename)
        if match:
            params = match.groupdict()
        else:
            raise ValueError(f'Filename format not recognized: {filename}')

        # Get data from the loaded pickle file
        d_list = data['d_list']
        log_err_rate = data['log_err_rate']

        # Add data rows for the DataFrame
        for id, d in enumerate(d_list):
            for ierr_rate, phys_err_rate in enumerate(data['phys_err_rate_list']):
                rows.append({
                    'd': d,
                    'phys_err_rate': phys_err_rate,
                    'log_err_rate': log_err_rate[id, ierr_rate],
                    **params  # Include all extracted parameters
                })

# Convert the collected rows into a DataFrame
df = pd.DataFrame(rows)

# Display the first few rows of the DataFrame for verification
print(df.head())

# Create a color map for different code sizes
unique_d_list = sorted(df['d'].unique())
color_map = cm.get_cmap('viridis', len(unique_d_list))

# Choose which column to use for color and linestyle
color_column = 'd'  # This can be 'd' or another column
linestyle_column = 'vortex_location'  # This can be 'vortex_location' or another column
marker_column = 'logical_operator_direction'  # This can be 'logical_operator_direction' or another column

# Create a new plot
plt.figure()

# Filter the DataFrame to only include X logical operators and vortex_sign = '1'
df = df.query('logical_operator_pauli_type == "X" and vortex_sign == "1" and noise_type == "DEPOLARIZE1"')

# Group by the chosen columns and plot each group
for (color_val, linestyle_val, marker_val), group in df.groupby([color_column, linestyle_column, marker_column]):
    linestyle = '--' if linestyle_val == 'x' else '-'  # Dashed for vortex='x', solid for None
    color = color_map(unique_d_list.index(color_val))  # Color based on code size 'd'
    marker = 'x' if marker_val == 'x' else 's'  # Marker for logical_operator_direction

    # Ensure data is sorted by phys_err_rate for proper line plotting
    group = group.sort_values(by='phys_err_rate')

    plt.errorbar(group['phys_err_rate'], group['log_err_rate'],
                 np.sqrt(group['log_err_rate'] * (1 - group['log_err_rate']) / 100000),
                 # Replace shots=100000 if needed
                 label=f'{color_val}, {linestyle_val}, {marker_val}',
                 linestyle=linestyle, color=color, marker=marker)  # Add marker for clarity

# Customize the plot
plt.xlabel('physical error rate')
plt.ylabel('logical error rate')
plt.yscale('log')
plt.xscale('log')
plt.legend(title=f'{color_column}, vortex_direction, {marker_column}', ncol=2)
edit_graph('Physical Error Rate', 'Logical Error Rate',
           scale=1.5)

# Show the plot
plt.savefig('figures/threshold.pdf')


plt.show()
