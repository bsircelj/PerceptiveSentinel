import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def save_figure(plt, file_name):
    plt.savefig(f'Results/{file_name}', dpi=300, bbox_inches='tight')


def plot_corr(corr, corr_labels, size=6, file_name=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(size, size)
    im = plt.imshow(corr)
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    corr_ticks = range(len(corr_labels))
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(corr_labels)))
    ax.set_yticks(np.arange(len(corr_labels)))
    ax.set_xticklabels(corr_labels)
    ax.set_yticklabels(corr_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha='left', va='center', rotation_mode='anchor',
             horizontalalignment='left',
             verticalalignment='top')
    if file_name:
        save_figure(plt, file_name)


def cluster_df(df, k=0.5):
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, k * d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    return df


# Correlation matrix
corr = i_filtered.corr()
# Plot correlation matrix.
plot_corr(corr, i_filtered.columns, size=16, file_name='correlation_matrix_basic.png')
# Plot clustered correlation matrix.
dsc = cluster_df(i_filtered)
plot_corr(dsc.corr(), dsc.columns, size=16, file_name='correlation_matrix_basic_clusterred.png')
dsc2 = cluster_df(i_filtered, k=0.02)
plot_corr(dsc2.corr(), dsc2.columns, size=16, file_name='correlation_matrix_basic_clusterred2.png')
