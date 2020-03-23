import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, embeddings, domains, labels, method='tsne'):
        assert method in ['tsne', 'pca'], 'method should be one of: "tsne", "pca"'
        
        embeddings = np.asarray(embeddings)
        assert embeddings.ndim == 2
        
        domains = np.asarray(domains)
        assert domains.ndim == 1 and domains.shape[0] == embeddings.shape[0]
        
        labels = np.asarray(labels)
        assert labels.ndim == 1 and labels.shape[0] == embeddings.shape[0]
        
        pass
    
    def visualize(self, figsize):
        assert isinstance(figsize, tuple) and len(figsize) == 2
        
        pass
