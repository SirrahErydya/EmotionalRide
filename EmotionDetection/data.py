import os
import csv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class FER2013(Dataset):
    def __init__(self, root, train=True, transform=None, reduce_emotions=False):
        self.train = train
        if self.train:
            path = os.path.join(root, 'train.csv')
        else:
            path = os.path.join(root, 'test.csv')
        self.transform = transform
        self.data = []
        self.targets = []
        self.reduce_emotions = reduce_emotions
        self.make_dataset(path)

    def __getitem__(self, item):
        img = self.data[item]
        if self.train:
            return img, self.targets[item]
        return img

    def __len__(self):
        return len(self.data)

    def make_dataset(self, path):
        with open(path) as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                image = np.fromstring(row['pixels'], dtype=np.float32, sep=' ').reshape((48, 48))
                if self.train:
                    emotion = np.int32(row['emotion'])
                    if self.reduce_emotions:
                        if emotion > 0:
                            emotion = emotion - 1
                        assert emotion < 6
                    self.targets.append(emotion)
                    if self.transform is not None:
                        self.targets.append(emotion)
                self.data.append(ToTensor()(image))
                if self.transform is not None:
                    self.data.append(self.transform(image))


class CKPlus(Dataset):
    def __init__(self, root, transform=None, reduce_emotions=False):
        self.path = os.path.join(root, "CK+48")
        self.transform = transform
        self.data = []
        self.targets = []
        self.reduce_emotions = reduce_emotions
        self.emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']
        for i in range(len(self.emotions)):
            self.get_data(i)

    def __getitem__(self, item):
        img = self.data[item]
        return img, self.targets[item]

    def __len__(self):
        return len(self.data)

    def get_data(self, target):
        img_folder = os.path.join(self.path, self.emotions[target])
        if self.reduce_emotions and target > 0:
            target = target - 1
        for img_file in os.listdir(img_folder):
            if img_file.endswith('.png'):
                pil_img = Image.open(os.path.join(img_folder, img_file))
                img = np.array(pil_img, dtype=np.float32)
                self.data.append(ToTensor()(img))
                self.targets.append(np.int32(target))
                if self.transform is not None:
                    self.data.append(self.transform(img))
                    self.targets.append(np.int32(target))


class CombiDataset(Dataset):
    def __init__(self, fer_root, ck_root, transform=None, reduce_emotions=False):
        self.transform = transform
        fer = FER2013(fer_root, transform=transform, reduce_emotions=reduce_emotions)
        ck = CKPlus(ck_root, transform=transform, reduce_emotions=reduce_emotions)
        self.data = fer.data + ck.data
        print("Combined dataset: FER2013 and CKPlus")
        print("FER Samples:", len(fer.data))
        print("CKPlus samples:", len(ck.data))
        self.targets = fer.targets + ck.targets

    def __getitem__(self, item):
        img = self.data[item]
        return img, self.targets[item]

    def __len__(self):
        return len(self.data)