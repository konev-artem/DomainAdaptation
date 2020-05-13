from domainadaptation.experiment import DANNExperiment
import os
import logging


if __name__ == '__main__':
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    domain_list = ['Art', 'Clipart', 'Product', 'Real World']

    dataset_path = '/data/ensirkiza/office_home/'
    dataset_type = 'resnet50'
    weights = 'imagenet'
    pooling_type = 'max'
    num_trainable_layers = 10
    epochs = 60
    img_size = (224, 224)
    num_classes = 65
    batch_size = 16
    lr = 3e-4

    for source_domain in domain_list:
        for target_domain in domain_list:
            if source_domain == target_domain:
                continue
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
                'learning_rate': lr,
                'epochs': epochs,
            }

            print("\n------WITH TARGET = ", target_domain, " ----WITH SOURCE = ", source_domain, '\n')
            dann_experiment = DANNExperiment(config)
            dann_experiment()
            print("\n----------------------------------------------------------------------------\n")
