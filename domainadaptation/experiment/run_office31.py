from domainadaptation.experiment.dann import DANNExperimentOffice31

import logging
logging.getLogger('tensorflow').disabled = True

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run on office-31')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to dataset root (default: ".")')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = {
        'backbone': {
            'type': 'resnet50',
            'num_trainable_layers': 7,
            'img_size': (224, 224),
            'weights': 'imagenet',
            'pooling': 'max'},
        'dataset': {
            'classes': 31,
            'path': args.dataset_root,
            'augmentations': {},
            'source': "amazon",
            'target': 'webcam'},
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 1e-4,
        'grads_update_freq': 1,
    }

    # config['dataset']['source'] = 'dslr'
    # config['dataset']['target'] = 'webcam'
    print(f"Run {config['dataset']['source']} -> {config['dataset']['target']}")
    exp = DANNExperimentOffice31(config=config)
    exp(train_domain_head=True)

    config['dataset']['source'] = 'webcam'
    config['dataset']['target'] = 'dslr'
    print(f"Run {config['dataset']['source']} -> {config['dataset']['target']}")
    exp = DANNExperimentOffice31(config=config)
    exp(train_domain_head=True)
