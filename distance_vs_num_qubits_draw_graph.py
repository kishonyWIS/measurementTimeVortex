import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('distance_vs_vortices.csv') #_by_equation

# each row appears twice with column 'logical_op_direction' is 'x' or 'y' and dist differs between the two rows
# merge the two rows into one row with columns 'dist_x' and 'dist_y'

def is_allowed_vortex_numbers(Lx, Ly, vx, vy, discount=0):
    condition1 = -vy - (3*Ly - 2)/(2*Lx - 1) * abs(vx) + 2*Ly - 1#-vy - (3*Ly - 2)/(2*Lx - 1) * abs(vx) + 2*Ly - 1
    condition2 = vy + Ly
    print(f'condition1={condition1}, condition2={condition2}')
    return condition1>=-discount and condition2>0

# filter out the rows that do not satisfy the condition
df = df[df.apply(lambda row: is_allowed_vortex_numbers(row['Lx'], row['Ly'], row['vx'], row['vy'], discount=1), axis=1)]

df_x = df[df['logical_op_direction'] == 'x'].drop(columns='logical_op_direction').rename(columns={'dist': 'dist_x'})
df_y = df[df['logical_op_direction'] == 'y'].drop(columns='logical_op_direction').rename(columns={'dist': 'dist_y'})

# Merge the two DataFrames on all columns except for 'dist_x' and 'dist_y'
merge_columns = [col for col in df_x.columns if col not in ['dist_x', 'dist_y']]
df = pd.merge(df_x, df_y, on=merge_columns)

# Create a new column 'dist' that is the minimum of 'dist_x' and 'dist_y'
df['dist'] = np.minimum(df['dist_x'], df['dist_y'])

# plot the distance vs number of qubits
plt.figure()
plt.scatter(df['n_qubits'], df['dist'])
plt.xlabel('Number of qubits')
plt.ylabel('Distance')



plt.title('(vx,vy,Lx,Ly)')

# for each number of qubits, print (vx,vy,Lx,Ly) for all points with maximal distance
for i, row in df.groupby('n_qubits').apply(lambda x: x.loc[x['dist'].idxmax()]).iterrows():
    print(f'Number of qubits: {row["n_qubits"]}, maximal distance: {row["dist"]}')
    df_maximal = df[df['n_qubits'] == row['n_qubits']][df['dist'] == row['dist']]
    print(df_maximal)
    # row is the row with the minimal sum of absolute value of vx and vy
    row = df_maximal.loc[(df_maximal['vx'].abs() + df_maximal['vy'].abs()).idxmin()]
    plt.text(row['n_qubits'], row['dist'], f'({row["vx"]},{row["vy"]},{row["Lx"]},{row["Ly"]})')

# plot in red the points for vx = vy = 0
df_zero = df[df['vx'] == 0][df['vy'] == 0]
plt.scatter(df_zero['n_qubits'], df_zero['dist'], color='red')

plt.show()