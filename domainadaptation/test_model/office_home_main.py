import sys
sys.path.append('./..')
sys.path.append('./../..')

from experiment import DANNExperiment

if __name__ == '__main__':
    dataset_path = '../../office_home/'
    dataset_type = 'resnet101'
    source_domain = 'Art'
    target_domain = 'Real World'
    epochs = 2
    img_size = (224, 224)
    num_classes = 65
    batch_size = 32

    config = {
        'backbone':
            {
                'type': dataset_type,
                'img-size': img_size,
            },
        'dataset':
            {
                'path': dataset_path,
                'classes': num_classes,
                'source': source_domain,
                'target': target_domain
            },
        'batch_size': batch_size,
        'epochs': epochs,
    }
    dann_experiment = DANNExperiment(config)
    dann_experiment()
