import ast
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D

from plot_utils import *
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def tuple_int_converter(value):
    if not value or pd.isna(value):  # Check for empty or NaN values
        return ()  # Return an empty tuple
    try:
        # Handle square bracket format like `[ 2 -1]`
        if value.startswith("[") and value.endswith("]"):
            cleaned_value = value.strip("[]").split()
            return tuple(map(int, cleaned_value))
        # Handle parentheses format like `(2, -1)` if it appears
        elif value.startswith("(") and value.endswith(")"):
            return tuple(map(int, ast.literal_eval(value)))
        else:
            raise ValueError(f"Unsupported tuple format: {value}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid tuple format: {value}") from e


# Read the CSV, applying the converter to a specific column
df = pd.read_csv('data/data_threshold.csv', converters={'num_vortexes': tuple_int_converter, 'l1': tuple_int_converter, 'l2': tuple_int_converter})
df = df.drop(columns=['Unnamed: 0'])

# Draw the logical error rate "both" vs the system size for each error rate. Use full lines for no vortexes and dashed lines for 2L-1 vortexes.

df = df.drop(columns=['detectors', 'logical_operator_pauli_type', 'reps_with_noise'])
df = df[df['logical_operator_direction'] == 'both'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_both'})


# Draw the logical error rate 'both' vs. physical error rate for each system size. Use full lines for no vortexes and dashed lines with vortexes.
# Color according to square root of number of qubits

df['has_vortices'] = df['num_vortexes'].apply(lambda x: x != (0, 0))

# num_qubits = 2 * |L1 x L2|
df['l1_1'] = df['l1'].apply(lambda x: x[0])
df['l1_2'] = df['l1'].apply(lambda x: x[1])
df['l2_1'] = df['l2'].apply(lambda x: x[0])
df['l2_2'] = df['l2'].apply(lambda x: x[1])
df['num_qubits'] = 2 * abs(df['l1_1'] * df['l2_2'] - df['l1_2'] * df['l2_1'])

df = df[df['num_qubits'] < 100]


# for each num_qubits, distance, if there are multiple L1, L2, num_vortexes, choose the one with the lowest logical error rate at
phys_err_rate = 0.0031622776601683794
df_fixed_phys_err_rate = df[np.abs(df['phys_err_rate'] - 0.0031622776601683794) < 1e-10]
L1, L2, num_vortexes = [], [], []
for (num_qubits, has_vortices), group_L in df_fixed_phys_err_rate.groupby(['num_qubits', 'has_vortices']):
    min_logical_error_rate = group_L['log_err_rate_both'].min()
    min_logical_error_rate_group = group_L[group_L['log_err_rate_both'] == min_logical_error_rate]
    L1.append(min_logical_error_rate_group['l1'].iloc[0])
    L2.append(min_logical_error_rate_group['l2'].iloc[0])
    num_vortexes.append(min_logical_error_rate_group['num_vortexes'].iloc[0])
df = df.query('l1 in @L1 and l2 in @L2 and num_vortexes in @num_vortexes')


# Color based on the square root of num_qubits
df['sqrt_num_qubits'] = np.sqrt(df['num_qubits'])

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define colormap based on the square root values
num_qubits_to_color = {sqrt_num_qubits: color for sqrt_num_qubits, color in zip(sorted(df['sqrt_num_qubits'].unique()), plt.cm.viridis(np.linspace(0, 1, len(df['sqrt_num_qubits'].unique()))))}
colormap = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=min(df['sqrt_num_qubits']), vmax=max(df['sqrt_num_qubits']))

fig1, ax1 = plt.subplots()
for (l1, l2, num_vortexes, num_qubits, sqrt_num_qubits), group_L in sorted(df.groupby(['l1', 'l2', 'num_vortexes', 'num_qubits', 'sqrt_num_qubits']), key=lambda x: x[0][3]):
    has_vortexes = num_vortexes != (0, 0)
    num_qubits = group_L['num_qubits'].iloc[0]
    linestyle = '-' if not has_vortexes else ':'
    marker = 'o' if not has_vortexes else 'x'
    L1 = (l1[0], l1[1], -6 * num_vortexes[0])
    L2 = (l2[0], l2[1], -6 * num_vortexes[1])
    ax1.loglog(group_L['phys_err_rate'], group_L['log_err_rate_both'], linestyle=linestyle, marker=marker, color=num_qubits_to_color[sqrt_num_qubits])

# Create the colorbar with the square root of num_qubits but display num_qubits on the colorbar
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Set array to associate with the ScalarMappable
cbar = fig1.colorbar(sm, ax=ax1, label='Number of qubits')
cbar.set_ticks(np.sqrt(sorted(np.unique(df['num_qubits']))))  # Set the ticks to the square root of the num_qubits values
cbar.set_ticklabels(sorted(np.unique(df['num_qubits'])))  # Set the tick labels to the num_qubits values

# Add legend
legend_handles = [
    Line2D([], [], marker='o', color='black', linestyle='-', label='Without Vortices'),
    Line2D([], [], marker='x', color='black', linestyle=':', label='With Vortices')
]
ax1.legend(handles=legend_handles, handletextpad=0.4, borderpad=0.5, labelspacing=0.5, fontsize=9.5, loc='upper left')

# Filter data for the second plot
df = df[np.abs(df['phys_err_rate'] - 0.0031622776601683794) < 1e-10]

fig_mask = patches.Rectangle(
    (0.5, 0.19),  # Lower-left corner of the rectangle
    0.3,          # Width
    0.352,         # Height
    transform=fig1.transFigure,
    facecolor='white',  # Specify fill color
    edgecolor='#b0b0b0',   # Specify border color
    linewidth=1,        # Set the thickness of the border
    zorder=4,           # Ensure it is rendered below the inset
    alpha=1             # Set opacity
)
fig1.patches.append(fig_mask)

# Add inset axes to fig1
ax2 = fig1.add_axes([0.6, 0.285, 0.18, 0.25], zorder=5)
for with_vortices, group_L in df.groupby('has_vortices'):
    linestyle = '-' if not with_vortices else ':'
    marker = 'o' if not with_vortices else 'x'
    ax2.semilogy(
        group_L['num_qubits'],
        group_L['log_err_rate_both'],
        linestyle=linestyle,
        marker=marker,
        color='k',
        label='with vortices' if with_vortices else 'without vortices'
    )

edit_graph('Number of qubits', 'Logical error rate', ax=ax2, scale=1)


# Edit and save the main figure
edit_graph('Physical error rate', 'Logical error rate', ax=ax1, scale=1.5)
plt.savefig('figures/threshold_graph_with_inset.pdf')

plt.show()
