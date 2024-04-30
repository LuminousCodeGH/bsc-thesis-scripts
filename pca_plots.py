import matplotlib.pyplot as plt
import numpy as np

def pca_summary(pca_model):
    sdev = np.sqrt(pca_model.explained_variance_)
    var_cor = np.apply_along_axis(lambda c: c*sdev, 0, pca_model.components_)
    var_cos2 = var_cor**2
    comp_cos2 = np.sum(var_cos2, 1)
    var_contrib = np.apply_along_axis(lambda c: c/comp_cos2, 0, var_cos2)
    return var_cor, var_contrib

def pca_screeplot(pca_model, figsize=(5,3), title_add=''):
    """
    Shows a barplot of the fraction of variance explained by each PC.
    """
    var_cor, var_contrib = pca_summary(pca_model)
    dim = np.arange(var_cor.shape[1])
    var = pca_model.explained_variance_ratio_

    plt.figure(figsize=figsize)
    plt.rc('axes', axisbelow=True)
    plt.grid(axis = 'y', linewidth = 0.5)
    plt.bar(dim, 100*var)
    plt.xticks(dim, dim+1, rotation=90)
    plt.xlabel('Component')
    plt.ylabel('Explained variance (%)')
    plt.title(f'Scree plot {title_add}')
    plt.tight_layout()

def pca_corplot(pca_model, column_names, comp = [0,1], figsize=(5,5), title_add='', vec_filter: list[str | bool]=None):
    """
    Shows the variables in a correlation circle.
    The projection of each variable on a PC represents its correlation with that PC.
    """
    var_cor, var_contrib = pca_summary(pca_model)
    
    plt.figure(figsize=figsize)
    plt.rc('axes', axisbelow=True)
    plt.grid(linewidth = 0.5)
    plt.axhline(linestyle='--', color='k')
    plt.axvline(linestyle='--', color='k')
    plt.gcf().gca().add_patch(plt.Circle((0,0),1,color='grey',fill=False))
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel(f'PC{comp[0]+1}')
    plt.ylabel(f'PC{comp[1]+1}')
    plt.title(f'Correlation circle PC{comp[0]+1}-PC{comp[1]+1} {title_add}')
    plt.tight_layout()

    for i in range(pca_model.n_components_):
        if vec_filter is not None:
            if isinstance(vec_filter[i], str):
                if vec_filter[i] not in column_names:
                    continue
            elif isinstance(vec_filter[i], bool):
                if not vec_filter[i]:
                    continue
        x = var_cor[comp[0],i]
        y = var_cor[comp[1],i]
        plt.arrow(0,0,x,y, color='k',
                 head_length=.025, head_width=.025, length_includes_head=True)
        plt.text(x,y,
                 horizontalalignment='left' if x>0 else 'right',
                 verticalalignment='bottom' if y>0 else 'top',
                 color='k', s=list(column_names)[i])

def pca_contribplot(pca_model, column_names, comp = 0, figsize=(6,5), title_add=''):
    """
    Shows a barplot of the contribution of each variable to a PC.
    """
    var_cor, var_contrib = pca_summary(pca_model)
    dim = np.arange(var_cor.shape[1])
    i = (-var_contrib[comp]).argsort()

    plt.figure(figsize=figsize)
    plt.rc('axes', axisbelow=True)
    plt.grid(axis = 'y', linewidth = 0.5)
    plt.bar(dim, 100*var_contrib[comp][i])
    plt.axhline(y=100/len(dim), linestyle='--', color='r')
    plt.xticks(dim, column_names[i], rotation=90)
    plt.ylabel('Contribution (%)')
    plt.title(f'Contribution of variables to PC{comp+1} {title_add}')
    plt.tight_layout()

def pca_biplot(X, pca_model, column_names, comp = [0,1], clusters = None, labels = None, figsize=(10,8), title_add='', vec_filter: list[str | bool]=None):
    """
    Shows a scatterplot showing each observation in the 2D space defined by two PCs,
    along with the vectors corresponding to each feature.
    Observations can optionally be identified by a label and a color (eg. to identify clusters).
    Adapted from https://jbhender.github.io/Stats506/F17/Projects/G18.html
    """
    xvector = pca_model.components_[comp[0]]
    yvector = pca_model.components_[comp[1]]

    xs = pca_model.transform(X)[:,comp[0]]
    ys = pca_model.transform(X)[:,comp[1]]
        
    plt.figure(figsize=figsize)
    plt.rc('axes', axisbelow=True)
    plt.grid(linewidth = 0.5)
    plt.axhline(linestyle='--', color='k')
    plt.axvline(linestyle='--', color='k')
    plt.xlim([min([0, min(xs), min(xvector*max(xs)*1.1)]),
              max([0, max(xs), max(xvector*max(xs)*1.1)])
             ])
    plt.ylim([min([0, min(ys), min(yvector*max(ys)*1.1)]),
              max([0, max(ys), max(yvector*max(ys)*1.1)])
             ])
    plt.xlabel(f'PC{comp[0]+1}')
    plt.ylabel(f'PC{comp[1]+1}')
    plt.title(f'Biplot PC{comp[0]+1}-PC{comp[1]+1} {title_add}')
    
    # plot vectors
    for i in range(len(xvector)):
        if vec_filter is not None:
            if isinstance(vec_filter[i], str):
                if vec_filter[i] not in column_names:
                    continue
            elif isinstance(vec_filter[i], bool):
                if not vec_filter[i]:
                    continue
        x = xvector[i]*max(xs)
        y = yvector[i]*max(ys)
        plt.arrow(0, 0, x, y, color='k',
                 head_length=.1, head_width=.1, length_includes_head=True)
        plt.text(x*1.1, y*1.1, color='k', s=list(column_names)[i])

    # plot observations
    if labels is None:
        labels = np.arange(X.shape[0]).astype('str')
    if clusters is None:
        clusters = np.zeros(X.shape[0], dtype=int)
    colors = plt.colormaps['tab10'].colors
    color_map = {}
    for c in np.unique(clusters):
        color_map[c] = len(color_map)
        plt.plot(0,0, color = colors[color_map[c]], label=str(c)) # only to make sure the legend appears
    plt.legend()
    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], color = colors[color_map[clusters[i]]], marker='.')
        plt.text(xs[i], ys[i], color = colors[color_map[clusters[i]]], label=str(clusters[i]),
                 s = labels[i], ha='center', va='top' if ys[i]>0 else 'bottom')