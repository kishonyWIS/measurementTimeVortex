import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure Seaborn style
sns.set(style="whitegrid")

def set_latex_params(scale=1.0):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 10 * scale,  # Base font size
        'axes.labelsize': 10 * scale,
        'axes.titlesize': 10 * scale,
        'xtick.labelsize': 8 * scale,
        'ytick.labelsize': 8 * scale,
        'legend.fontsize': 8 * scale,
        'figure.titlesize': 12 * scale,
        'text.latex.preamble': r'''
            \usepackage{subfigure}
            \usepackage{amsmath, amssymb, amsfonts}
            \usepackage{xcolor}
            \definecolor{darkblue}{RGB}{0,0,150}
            \definecolor{nightblue}{RGB}{0,0,100}
            \usepackage{graphicx,mathtools,bm,bbm}
            \usepackage{MnSymbol}
            \usepackage[colorlinks,citecolor=darkblue,linkcolor=darkblue,urlcolor=nightblue]{hyperref}
            \usepackage[english]{babel}
            \usepackage[babel,kerning=true,spacing=true]{microtype}
            \usepackage[utf8]{inputenc}
            \usepackage{soul}
        '''
    })


def edit_graph(xlabel=None, ylabel=None, ax=None, title=None, legend_title=None,
               colormap=None, colorbar_title=None, colorbar_args={}, tight=True,
               ylabelpad=None, colorbar_xticklabels=None, colorbar_yticklabels=None,
               xticks=None, yticks=None, xticklabels=None, yticklabels=None,
               scale=1.0):
    set_latex_params(scale)

    if ax is None:
        ax = plt.gca()  # Get current axis if not provided

    # Apply labels and title if specified
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10 * scale)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10 * scale, labelpad=ylabelpad if ylabelpad is not None else None)
    if title is not None:
        ax.set_title(title, fontsize=12 * scale)
    if legend_title is not None:
        ax.legend(title=legend_title, fontsize=8 * scale)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    plt.tick_params(axis='both', which='major', labelsize=10 * scale)

    # Apply colormap if specified
    if colormap is not None:
        for im in ax.get_images():
            im.set_cmap(colormap)
        if 'colorbar' not in colorbar_args:
            # Create and configure colorbar if it’s not passed
            colorbar = plt.colorbar(ax.images[-1], ax=ax, **colorbar_args)
            if colorbar_title is not None:
                colorbar.set_label(colorbar_title, fontsize=8 * scale)
            if colorbar_xticklabels is not None:
                colorbar.set_ticks(colorbar_xticklabels)
            if colorbar_yticklabels is not None:
                colorbar.set_ticks(colorbar_yticklabels)
            colorbar.ax.tick_params(labelsize=10 * scale)

    if tight:
        plt.tight_layout()
    plt.draw()  # Update the plot with new settings


# Optionally, include a test plot when the module is run directly
if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # First figure: imshow
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(np.random.rand(10, 10), cmap='viridis')  # Example image for colorbar
    # Apply formatting
    edit_graph(ax=ax1, colormap='plasma', colorbar_title='Colorbar Title', colorbar_args={'fraction': 0.046, 'pad': 0.04},
               ylabelpad=10, colorbar_xticklabels=[0, 0.5, 1], colorbar_yticklabels=[0, 0.5, 1])

    # Second figure: sin(x) plot
    fig2, ax2 = plt.subplots()
    ax2.plot(x, y, label='$sin(x)$')
    # Apply formatting
    edit_graph(xlabel='$k_xa$', ylabel='y', ax=ax2, title='$sin(x)$', legend_title='Legend Title', scale=1.0)
    plt.show()
