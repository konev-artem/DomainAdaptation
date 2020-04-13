import os
import logging

from domainadaptation.experiment import DANNExperiment

if __name__ == '__main__':

    # disable useless tf warnings
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    exp = DANNExperiment(config={
        'backbone': {
            'type': 'resnet50',
            'num_trainable_layers': 10,
            'img-size': (224, 224),
            'weights': 'imagenet',
            'pooling': 'max'
        },
        'dataset': {
            'classes': 12,
            'path': '/data/eikolodin/visda',
            'augmentations': {},
            'source': 'train',
            'target': 'validation'
        },
        'clip_grads': -1,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'epochs': 10,
    })

    exp.experiment_domain_adaptation_v2()
