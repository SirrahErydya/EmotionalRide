import os
import csv
from torch.utils.data import Dataset
import numpy as np


class FER2013(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        if self.train:
            path = os.path.join(root, 'train.csv')
        else:
            path = os.path.join(root, 'test.csv')
        self.transform = transform
        self.data = []
        self.targets = []
        self.make_dataset(path)
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __getitem__(self, item):
        img = self.data[item]
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            return img, self.targets[item]
        return img

    def __len__(self):
        return len(self.data)

    def make_dataset(self, path):
        with open(path) as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                image = np.fromstring(row['pixels'], dtype=np.float32, sep=' ').reshape((1, 48, 48))
                if self.train:
                    self.targets.append(np.int32(row['emotion']))
                self.data.append(image)

