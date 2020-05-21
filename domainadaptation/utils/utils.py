import numpy as np

def get_features_and_labels(model, generator, count):
    features = []
    labels = []
    for i in range(count):
        try:
            X, y = next(generator)
        except:
            break
        if len(y.shape) == 2:
            y = np.argmax(y, axis=-1)
        features.append(model(X))
        labels.append(y)

    return np.vstack(features), np.hstack(labels)