import pandas as pd

# Load datasets
data_with_vortices = pd.read_csv('data/torus_configurations_with_vortices.csv')
data_without_vortices = pd.read_csv('data/torus_configurations_without_vortices.csv')

# Add computed column R = N / D^2 and round to two decimal places
data_with_vortices['R'] = (data_with_vortices['# qubits'] / data_with_vortices['distance'] ** 2).round(2)
data_without_vortices['R'] = (data_without_vortices['# qubits'] / data_without_vortices['distance'] ** 2).round(2)

# Add a 'type' column to differentiate datasets
data_with_vortices['type'] = 'with'
data_without_vortices['type'] = 'without'

# Combine datasets, ensuring all D values are represented
all_distances = pd.concat([
    data_with_vortices[['distance']],
    data_without_vortices[['distance']]
]).drop_duplicates().sort_values(by='distance')

# Align rows for each distance value
aligned_rows = []
for distance in all_distances['distance']:
    with_row = data_with_vortices[data_with_vortices['distance'] == distance]
    without_row = data_without_vortices[data_without_vortices['distance'] == distance]

    max_rows = max(len(with_row), len(without_row))
    for i in range(max_rows):
        aligned_rows.append({
            'distance': distance if i == 0 else '',  # Only show D in the first row
            'without_N': without_row.iloc[i]['# qubits'] if i < len(without_row) else '-',
            'without_L1': without_row.iloc[i]['L1'] if i < len(without_row) else '-',
            'without_L2': without_row.iloc[i]['L2'] if i < len(without_row) else '-',
            'with_N': with_row.iloc[i]['# qubits'] if i < len(with_row) else '-',
            'with_L1': with_row.iloc[i]['L1'] if i < len(with_row) else '-',
            'with_L2': with_row.iloc[i]['L2'] if i < len(with_row) else '-',
        })

# Create DataFrame for aligned rows
aligned_data = pd.DataFrame(aligned_rows)

# Prepare the LaTeX table
latex_table = r"""
\begin{table}[ht]
\centering
\caption{Optimal embedding of the Floquet color code on the torus with and without time vortices.}
\begin{tabular}{c|ccc|ccc}
\toprule
\hline
\hline
& \multicolumn{3}{c|}{\textbf{Without Vortices}} & \multicolumn{3}{c}{\textbf{With Vortices}} \\
\midrule
\hline
\textbf{$D$} & \textbf{$N$} & \textbf{$\mathbf{L}_1$} & \textbf{$\mathbf{L}_2$} & \textbf{$N$} & \textbf{$\mathbf{L}_1$} & \textbf{$\mathbf{L}_2$} \\
\midrule
"""

# Add rows to the LaTeX table
for _, row in aligned_data.iterrows():
    latex_row = (
        f"{row['distance']} & {row['without_N']} & {row['without_L1']} & {row['without_L2']} & "
        f"{row['with_N']} & {row['with_L1']} & {row['with_L2']} \\\\"
    )
    latex_table += latex_row + "\n"

# Add footer
latex_table += r"""
\hline
\hline
\bottomrule
\end{tabular}
\label{tab: optimal}
\end{table}
"""

# Save or print the table
with open('table_aligned_with_placeholders.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table with aligned rows and grouped $D$ values saved as 'table_aligned_with_placeholders.tex'.")
