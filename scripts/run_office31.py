from domainadaptation.experiment import DANNExperiment

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run on office-31')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to dataset root (default: ".")')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    exp = DANNExperiment(config={
        'backbone': {
            'type': 'resnet50',
            'num_trainable_layers': 7,
            'img-size': (224, 224),
            'weights': 'imagenet',
            'pooling': 'max'},
        'dataset': {
            'classes': 31,
            'path': args.dataset_root,
            'augmentations': {},
            'source': "amazon",
            'target': 'webcam'},
        'batch_size': 16,
        'epochs': 10,
        'steps': 4000 // 16,
        'lr': 1e-4,
        'grads_update_freq': 1,
    })

    exp.experiment_domain_adaptation_v2()
