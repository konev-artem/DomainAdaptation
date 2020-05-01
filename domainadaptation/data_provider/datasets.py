import os

import numpy as np
from PIL import Image


class UnlabeledDataset:

    def __init__(self, root, img_size, store_in_ram):
        """
        Arguments:

            :param root: (str) path to root
            :param img_size: (int) to what size to resize the images
            :param store_in_ram: (bool) whether to store in RAM the data
        """

        self.img_size = img_size
        self.store_in_ram = store_in_ram
        self._dataset = []

        for cls_ in sorted(os.listdir(root)):
            for filename in sorted(os.listdir(os.path.join(root, cls_))):

                path = os.path.join(root, cls_, filename)

                if self.store_in_ram:
                    img = self._read_img(path=path, img_size=self.img_size)
                    self._dataset.append(img)
                else:
                    self._dataset.append(path)

    def __len__(self):

        return len(self._dataset)

    def __getitem__(self, idx):

        assert idx < len(self)

        img = self._dataset[idx] if self.store_in_ram else \
            self._read_img(self._dataset[idx], self.img_size)

        return img

    @staticmethod
    def _read_img(path, img_size):
        """
        Reads image from path, resizes to (img_size x img_size)
        and convert to numpy array

        Arguments:

            :param path: (str) path to image
            :param img_size: (int) to what size to resize the images
        """

        img = Image.open(path).resize((img_size, img_size))
        return np.array(img)


class LabeledDataset(UnlabeledDataset):

    def __init__(self, root, img_size, store_in_ram, type_label, cluster_labels=None):
        """
        Arguments:

            :param root: (str) path to root
            :param img_size: (int) to what size to resize the images
            :param store_in_ram: (bool) whether to store in RAM the data
            :param type_label: (int) 0 if supervised labeling is performed, 1 if cluster labeling
            :param cluster_labels: (optional) cluster labels in case of type_label == 1
        """

        super().__init__(root, img_size, store_in_ram)

        self._targets = []

        if type_label == 0:
            self._supervised_labeling(root)
        else:
            assert cluster_labels is not None

            self._targets = cluster_labels

        self._targets = np.asarray(self._targets)
        assert len(self._targets) == len(self)

        self.class_to_mask = {class_: np.asarray(self._targets == class_, dtype=np.int8)
                              for class_ in range(self._targets.max() + 1)}

    def _supervised_labeling(self, root):

        classes = sorted(os.listdir(root))
        class_map = dict(zip(classes, range(len(classes))))

        for cls_ in classes:
            len_dir = len(os.listdir(os.path.join(root, cls_)))
            self._targets.extend([class_map[cls_]] * len_dir)

    def __getitem__(self, idx):

        img = super().__getitem__(idx)

        return img, self._targets[idx]
