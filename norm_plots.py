import matplotlib.pyplot as plt
import anndata as ad
import seaborn as sns
import pandas as pd
import numpy as np
from colorsys import hsv_to_rgb


def _get_distinct_colors(n: int):
    colors = []
    for h in np.linspace(0.0, 1, n+1):
        colors.append(hsv_to_rgb(h, 0.7, 0.8))
    return colors[:-1]


def generate_boxplot_old(adata: ad.AnnData, layer: str) -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    bplot = ax.boxplot(_df.T, notch=True, labels=col_labels)
    plt.xticks(rotation=90)
    ax.set_title(f'Box Plot for {layer}')
    fig.show()
    del(_df)


def generate_boxplot(adata: ad.AnnData, layer: str) -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    ax = sns.boxplot(_df.T, orient='v', notch=True)
    ax.tick_params('x', rotation=90, width=5)
    plt.xticks(rotation=90)
    ax.set_title(f'Box Plot for {layer}')
    plt.show()
    del(_df)


def generate_densityplot(adata: ad.AnnData, layer: str, show_legend:bool=False) -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    sns.set_style('whitegrid')
    density = sns.kdeplot(_df, bw_method=0.5, legend=show_legend)
    density.set_title(f'Density Plot for {layer}')
    plt.show()
    del(_df)


def generate_clustermap(adata: ad.AnnData, layer: str, color_by: str='cogdx') -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    palette = dict( zip( adata.obs[color_by].unique(), _get_distinct_colors( len(adata.obs[color_by].unique()) ) ) )  # Does not work
    row_colors = adata.obs[color_by].map(palette)
    sns.clustermap(_df, row_colors=row_colors)
    plt.title(f'Clustermap for {layer}')
    plt.show()
    del(_df)


def generate_heatmap(adata: ad.AnnData, layer: str) -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    heatmap = sns.heatmap(_df)
    heatmap.set_title(f'Heatmap for {layer}')
    plt.show()


def generate_normalization_summary(adata: ad.AnnData, layer: str) -> None:
    col_labels = adata.var_names

    if layer == 'X':
        _df = pd.DataFrame(adata.X, columns=col_labels)
    else:
        _df = pd.DataFrame(adata.layers[layer], columns=col_labels)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,20))
    sns.set_style('whitegrid')
    density = sns.kdeplot(_df, bw_method=0.5, legend=False, ax=ax1)
    density.set_title(f'Density Plot for {layer}')
    boxplot = sns.boxplot(_df.T, orient='v', notch=True, ax=ax2)
    boxplot.tick_params('x', rotation=90, width=5)
    boxplot.set_title(f'Box Plot for {layer}')
    heatmap = sns.heatmap(_df,  ax=ax3)
    heatmap.set_title(f'Heatmap for {layer}')
    fig.show()