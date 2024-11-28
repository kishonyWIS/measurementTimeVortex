import ast
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import title

from plot_utils import *

def parse_tuple(value):
    return ast.literal_eval(value)

# Read the CSV, applying the converter to a specific column
df = pd.read_csv('data/data_threshold.csv', converters={'num_vortexes': parse_tuple})
df = df.drop(columns=['Unnamed: 0'])

# draw the logical error rate "both" vs the system size for each error rate. Use full lines for no vortexes and dashed lines for 2L-1 vortexes.

df = df.drop(columns=['detectors', 'logical_operator_pauli_type', 'reps_with_noise'])

# filter only the "both"
df_both = df[df['logical_operator_direction'] == 'both'].drop(columns='logical_operator_direction').rename(columns={'log_err_rate': 'log_err_rate_both'})

df_without_vortexes = df_both[df_both['num_vortexes'] == (0,0)]
df_with_vortexes = df_both[df_both['num_vortexes'] != (0, 0)]

df_fits = pd.DataFrame(columns=['vortexes', 'phys_err_rate', 'm', 'b'])
# Create a new plot
plt.figure()
for df, linestyle in zip([df_without_vortexes, df_with_vortexes], ['-', '--']):
    # reset colors
    plt.gca().set_prop_cycle(None)
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Group by the chosen columns and plot each group
    for name, group in df.groupby('phys_err_rate'):
        label = f'p={name}' if linestyle == '-' else None
        color = next(color_cycle)
        plt.semilogy(group['dx']/2, group['log_err_rate_both'], label=label, linestyle='', marker='o', color=color)
        # calculate a linear fit
        x = group['dx']/2
        y = np.log(group['log_err_rate_both'])
        # filter out the infinities
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]
        m, b = np.polyfit(x[-2:], y[-2:], 1)
        plt.semilogy(x, np.exp(m*x+b), linestyle=linestyle, color=color)
        df_fits = pd.concat([df_fits, pd.DataFrame({'vortexes': linestyle, 'phys_err_rate': [name], 'm': [m], 'b': [b]})])

plt.xlabel('L')
plt.ylabel('Logical error rate')
plt.legend()


# calculate the teraquop footprint for each error rate and vortexs vs. no vortexs
# extrapolate the logical error rate for the teraquop footprint

plt.figure()
for linestyle in ['-', '--']:
    df_fits_vortexes = df_fits[df_fits['vortexes'] == linestyle]
    phys_err_rate = []
    teraquop_footprint = []
    for row in df_fits_vortexes.itertuples():
        # calculate the teraquop footprint
        L = (np.log(1e-12) - row.b) / row.m
        dx = 2*L
        dy = 3*L
        N = dx*dy*4
        teraquop_footprint.append(N)
        phys_err_rate.append(row.phys_err_rate)
    plt.loglog(phys_err_rate, teraquop_footprint, linestyle=linestyle, marker='o')
plt.xlabel('Physical error rate')
plt.ylabel('Teraquop footprint')
plt.legend(['0 vortexes', '2L-1 vortexes'])




scale=1.8


# plot the logical error rate vs the physical error rate for each system size. Use full lines for no vortexes and dashed lines for 2L-1 vortexes.
plt.figure()
for df, linestyle in zip([df_without_vortexes, df_with_vortexes], ['-', '--']):
    # reset colors
    plt.gca().set_prop_cycle(None)
    # Group by the chosen columns and plot each group
    for name, group in df.groupby('dx'):
        label = str(int(name/2))+', '+str((0,0)) if linestyle == '-' else str(int(name/2))+', '+str((0,2*int(name/2)-1))
        if label:
            plt.loglog(group['phys_err_rate'], group['log_err_rate_both'], label=label, linestyle=linestyle, marker='o')
        else:
            plt.loglog(group['phys_err_rate'], group['log_err_rate_both'], linestyle=linestyle, marker='o')


plt.legend(title='$L, (n_x,n_y)$', fontsize=6*scale, title_fontsize=8*scale, ncols=2)



edit_graph('Physical error rate', 'Logical error rate', scale=scale)

plt.savefig('figures/threshold_graph.pdf')



plt.show()
