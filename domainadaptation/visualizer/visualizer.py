import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Visualizer:
    def __init__(self,
                 embeddings,                    # embeddings to be visualized
                                                # shape (n, dim)
                 
                 domains,                       # name of the domain for each embedding
                                                # shape (n,)
                 
                 labels,                        # name of the label/class for each embedding 
                                                # shape (n,)
                 
                 method='tsne',                 # dimensionality reduction method
                                                # pca and tsne are supported
                 
                 markers=['.', '*', 'X', 'd'],  # markers to scatter different domains
                                                # first N_DOMAINS markers are used
                                                # where N_DOMAINS is the number of different domains
                                                # more markers available at
                                                # https://matplotlib.org/3.1.1/api/markers_api.html
                 
                 colors=['black',               # colors to scatter different labels
                         'brown',               # first N_LABELS colors are used
                         'red',                 # first N_LABELS colors are used
                         'yellow',              # where N_LABELS is the number of different labels
                         'green',               # https://matplotlib.org/3.1.1/api/colors_api.html
                         'lightseagreen',
                         'aqua',
                         'dodgerblue',
                         'darkblue',
                         'blue',
                         'orchid',
                         'hotpink'],
                 
                 use_css_colors=False,
                 

                 **kwargs                       # arguments to be passed into T-SNE/PCA constructor
                                                # for example you n_jobs for T-SNE
                ):
        
        embeddings = np.asarray(embeddings)
        assert embeddings.ndim == 2, 'wrong embeddings shape'
        
        assert method in ['tsne', 'pca'], 'method should be one of: "tsne", "pca"'
        if method == 'tsne':
            self.embeddings_transformed = TSNE(**kwargs).fit_transform(embeddings)
        elif method == 'pca':
            self.embeddings_transformed = PCA(**kwargs).fit_transform(embeddings)
        
        self.domains = np.asarray(domains)
        assert domains.ndim == 1 and domains.shape[0] == embeddings.shape[0], 'wrong domains shape'
        
        self.labels = np.asarray(labels)
        assert labels.ndim == 1 and labels.shape[0] == embeddings.shape[0], 'wrong labels shape'
        
        assert len(markers) >= len(np.unique(domains)), 'not enough markers for domains'
        self.markers = markers
        
        self.colors = colors
        if use_css_colors:
            self.colors = list(CSS4_COLORS.keys())
        assert len(self.colors) >= len(np.unique(labels)), 'not enough colors for labels'
    
    def visualize(self,
                  *args,
                  title=None,         # string, title of plot
                  title_fontsize=18,  # title fontsize
                  figsize=(12, 8),    # size of plot,
                  alpha=1.0,          # alpha for scatter
                  size=50,            # size for scatter
                  draw_legend=True,   # draw legend or not
                  legend_fontsize=16, # legend fontsize
                  draw_ticks=True,    # draw x/y ticks or not
                  filename=None       # if not None, plot is saved as a picture into filename 
                 ):
        if args:
            raise ValueError("Function visualize accepts only named arguments")
        assert isinstance(figsize, tuple) and len(figsize) == 2
        
        plt.figure(figsize=figsize)
        if title is not None:
            plt.title(title, fontsize=title_fontsize)
        
        for domain_name, marker in zip(np.unique(self.domains), self.markers):
            domain_mask = self.domains == domain_name
            
            for label_name, color in zip(np.unique(self.labels), self.colors):
                label_mask = self.labels == label_name
                domain_and_label_mask = domain_mask & label_mask
                
                plt.scatter(
                    self.embeddings_transformed[domain_and_label_mask, 0],
                    self.embeddings_transformed[domain_and_label_mask, 1],
                    marker=marker,
                    color=color,
                    label='{} ({})'.format(label_name, domain_name),
                    alpha=alpha,
                    s=size)
        
        if draw_legend:
            plt.legend(fontsize=legend_fontsize)
            
        if not draw_ticks:
            plt.xticks([])
            plt.yticks([])

        if filename:
            plt.savefig(filename)
        
        plt.show()