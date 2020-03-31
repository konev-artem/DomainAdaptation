from domainadaptation.experiment import DANNExperiment
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == '__main__':
    dataset_path = '../../office_home/'
    dataset_type = 'resnet50'
    source_domain = 'Art'
    weights = 'imagenet'
    pooling_type = 'max'
    num_trainable_layers = 8
    target_domain = 'Real World'
    epochs = 2
    img_size = (224, 224)
    num_classes = 65
    batch_size = 32

    config = {
        'backbone':
            {
                'type': dataset_type,
                'num_trainable_layers': num_trainable_layers,
                'img_size': img_size,
                'weights': weights,
                'pooling': pooling_type
            },
        'dataset':
            {
                'path': dataset_path,
                'classes': num_classes,
                'source': source_domain,
                'target': target_domain,
                'augmentations': {},
            },
        'batch_size': batch_size,
        'epochs': epochs,
    }

    dann_experiment = DANNExperiment(config)
    dann_experiment.experiment_domain_adaptation()
