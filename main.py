import os
import logging

from domainadaptation.experiment import DANNExperiment

if __name__ == '__main__':

    # disable useless tf warnings
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    experiment = DANNExperiment(config={
        'backbone': {
            'type': 'resnet50',
            'num_trainable_layers': 10,
            'img_size': (224, 224),
            'weights': 'imagenet',
            'pooling': 'max'
        },
        'dataset': {
            'classes': 12,
            'path': '/data/jvchizh/datasets/visda',
            'augmentations': {},
            'source': 'train',
            'target': 'validation'
        },
        'visualizer': {
            'method': 'tsne'
        },
        'visualize': {
            'title': None,
            'title_fontsize': 18,
            'figsize': (12, 8),
            'alpha': 1.0,
            'size': 75,
            'draw_legend': False,
            'legend_fontsize': 16,
            'draw_ticks': True,
            'filename': './visualization/visda/resnet50_with_domain_adaptation'
        },
        'batch_size': 16,
        'learning_rate': 3e-4,
        'epochs': 2,
    })

    experiment(train_domain_head=True)
