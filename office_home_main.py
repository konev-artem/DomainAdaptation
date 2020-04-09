from domainadaptation.experiment import DANNExperiment
import os


if __name__ == '__main__':
    domain_list = ['Art', 'Clipart', 'Product', 'Real World']

    dataset_path = '/data/ensirkiza/office_home/'
    dataset_type = 'resnet50'
    weights = 'imagenet'
    pooling_type = 'max'
    num_trainable_layers = 8
    epochs = 60
    img_size = (224, 224)
    num_classes = 65
    batch_size = 16

    for source_domain in domain_list:
        for target_domain in domain_list:
            if source_domain == target_domain:
                continue
            config = {
                'backbone':
                    {
                        'type': dataset_type,
                        'num_trainable_layers': num_trainable_layers,
                        'img-size': img_size,
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
                'grads_update_freq': 1,
                'batch_size': batch_size,
                'epochs': epochs,
            }

            print("\n------WITH TARGET = ", target_domain, " ----WITH SOURCE = ", source_domain, '\n')
            dann_experiment = DANNExperiment(config)
            dann_experiment.experiment_domain_adaptation_v2()
            print("\n----------------------------------------------------------------------------\n")
