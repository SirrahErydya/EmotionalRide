import os
import csv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
REDUCED_INDICES = {0:0, 3:1, 4:2, 5:3}


class FerDataset(Dataset):
    def __init__(self, root, train=True, transform=None, reduce_emotions=False):
        self.train = train
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []
        self.reduce_emotions = reduce_emotions

    def __getitem__(self, item):
        img = self.data[item]
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            return img, self.targets[item]
        return img

    def __len__(self):
        return len(self.data)

    def make_dataset(self):
        raise NotImplementedError('Implement in Subclasses')


class FER2013(FerDataset):
    def __init__(self, root, train=True, transform=None, reduce_emotions=False):
        super(FER2013, self).__init__(root, train, transform, reduce_emotions)
        self.make_dataset()

    def make_dataset(self):
        if self.train:
            path = os.path.join(self.root, 'train.csv')
        else:
            path = os.path.join(self.root, 'test.csv')
        with open(path) as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                image = np.fromstring(row['pixels'], dtype=np.float32, sep=' ').reshape((48, 48))
                if self.train:
                    emotion = np.int32(row['emotion'])
                    if self.reduce_emotions:
                        try:
                            emotion = REDUCED_INDICES[emotion]
                        except KeyError:
                            continue
                    self.targets.append(emotion)
                self.data.append(image)


class CKPlus(FerDataset):
    def __init__(self, root, transform=None, reduce_emotions=False):
        super(CKPlus, self).__init__(root, transform=transform, reduce_emotions=reduce_emotions)
        self.emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.make_dataset()

    def make_dataset(self):
        for i in range(len(self.emotions)):
            emotion = i
            if self.reduce_emotions:
                try:
                    emotion = REDUCED_INDICES[i]
                except KeyError:
                    continue
            self.get_data(emotion)

    def get_data(self, target):
        path = os.path.join(self.root, "CK+48")
        img_folder = os.path.join(path, self.emotions[target])
        for img_file in os.listdir(img_folder):
            if img_file.endswith('.png'):
                pil_img = Image.open(os.path.join(img_folder, img_file))
                img = np.array(pil_img, dtype=np.float32)
                self.data.append(img)
                self.targets.append(np.int32(target))


class FacialExpression(FerDataset):
    def __init__(self, root, transform=None, reduce_emotions=False):
        super(FacialExpression, self).__init__(root, transform=transform, reduce_emotions=reduce_emotions)
        self.emo_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6, "contempt":7}
        self.make_dataset()

    def make_dataset(self):
        legend_path = os.path.join(self.root, 'data', 'legend.csv')
        img_path = os.path.join(self.root, 'images')
        with open(legend_path) as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                emotion = self.emo_map[row['emotion'].lower()]
                if self.reduce_emotions:
                    try:
                        emotion = REDUCED_INDICES[emotion]
                    except KeyError:
                        continue
                    self.targets.append(emotion)
                    pil_img = Image.open(os.path.join(img_path, row['image']))
                    pil_img = pil_img.resize((48,48), Image.BICUBIC)
                    pil_img = pil_img.convert('L')
                    img = np.array(pil_img, dtype=np.float32)
                    self.data.append(img)


class FERG(FerDataset):
    def __init__(self, root, transform=None, reduce_emotions=False):
        super(FERG, self).__init__(root, transform=transform, reduce_emotions=reduce_emotions)
        self.characters = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
        self.emo_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
        self.make_dataset()

    def make_dataset(self):
        for character in self.characters:
            char_path = os.path.join(self.root, character)
            for emotion in self.emo_map.keys():
                emo_idx = self.emo_map[emotion]
                if self.reduce_emotions:
                    try:
                        emo_idx = REDUCED_INDICES[emo_idx]
                    except KeyError:
                        continue
                folder = character + "_" + emotion
                full_path = os.path.join(char_path, folder)
                self.get_data(full_path, emo_idx)

    def get_data(self, path, target):
        for img_file in os.listdir(path):
            if img_file.endswith('.png'):
                pil_img = Image.open(os.path.join(path, img_file))
                pil_img = pil_img.resize((48,48), Image.BICUBIC)
                pil_img = pil_img.convert('L')
                img = np.array(pil_img, dtype=np.float32)
                self.data.append(img)
                self.targets.append(np.int32(target))


class CombiDataset(Dataset):
    def __init__(self, fer_root, ck_root, fe_root, ferg_root, transform=None, reduce_emotions=False):
        fer = FER2013(fer_root, transform=transform, reduce_emotions=reduce_emotions)
        ck = CKPlus(ck_root, transform=transform, reduce_emotions=reduce_emotions)
        fe = FacialExpression(fe_root, transform=transform, reduce_emotions=reduce_emotions)
        #ferg = FERG(ferg_root, transform=transform, reduce_emotions=reduce_emotions)
        if transform is None:
            self.data = fer.data + ck.data + fe.data #+ ferg.data
            self.targets = fer.targets + ck.targets + fe.targets #+ ferg.targets
        else:
            self.data = []
            self.targets = []
            max_length = np.max([len(fer.data), len(ck.data), len(fe.data), len(ferg.data)])
            for i in range(max_length):
                if i < len(fer):
                    data, target = fer[i]
                    self.data.append(data)
                    self.data.append(ToTensor()(fer.data[i]))
                    self.targets.append(target)
                    self.targets.append(target)
                if i < len(ck):
                    data, target = ck[i]
                    self.data.append(data)
                    self.data.append(ToTensor()(ck.data[i]))
                    self.targets.append(target)
                    self.targets.append(target)
                if i < len(fe):
                    data, target = fe[i]
                    self.data.append(data)
                    self.data.append(ToTensor()(fe.data[i]))
                    self.targets.append(target)
                    self.targets.append(target)
                if i < len(ferg):
                    data, target = ferg[i]
                    self.data.append(data)
                    self.data.append(ToTensor()(fe.data[i]))
                    self.targets.append(target)
                    self.targets.append(target)
        print("Combined dataset:")
        print("FER Samples:", len(fer.data))
        print("CKPlus samples:", len(ck.data))
        print("Facial expression samples:", len(fe.data))
        #print("FERG samples:", len(ferg.data))

    def __getitem__(self, item):
        img = self.data[item]
        return img, self.targets[item]

    def __len__(self):
        return len(self.data)