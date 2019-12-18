from os.path import join
import csv
import numpy as np
from tqdm import tqdm
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image



def load_annotations(path='/home/nick/Data/Datasets/oasis'):
    """
    Loads the OASIS annotations (per audio file)
    :param path:
    :return:
    """
    # Load arousal values
    with open(join(path, "OASIS.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        x = list(reader)

    header = x[0]
    filepaths = []
    labels = []

    for line in x[1:]:
        name, valence, arousal = line[1], float(line[4]), float(line[7])

        labels.append(np.asarray((arousal, valence)))
        cur_path = join(path, 'images', name.strip() + '.jpg')
        filepaths.append(cur_path)

    return filepaths, labels

class OASIS_Loader(Dataset):

    def __init__(self, dataset_path='/home/nick/Data/Datasets/oasis', transform=None, deploy=False, train=False):
        self.transform = transform

        filepaths, labels = load_annotations(dataset_path)
        self.filepaths = np.asarray(filepaths)
        self.labels = np.asarray(labels)
        mean = np.mean(self.labels, axis=0)
        self.labels = self.labels - mean

        # Shuffle the dataset
        idx = np.random.permutation(len(self.labels))

        if not deploy:
            if train:
                idx = idx[:700]
            else:
                idx = idx[700:]

        self.labels = self.labels[idx]
        self.filepaths = self.filepaths[idx]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.filepaths[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        # image = io.imread(img_name)

        label = self.labels[idx]
        label = label.astype('float').reshape(-1, 2)

        if self.transform:
            image = self.transform(image)

        return image, label



def get_oasis_dataset_loaders(batch_size=32):


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    cur_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), normalize,])

    train_dataset = OASIS_Loader(train=True, deploy=False, transform=cur_transforms)
    test_dataset = OASIS_Loader(train=False, deploy=False, transform=cur_transforms)
    deploy_dataset = OASIS_Loader(deploy=True, transform=cur_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    deploy_loader = torch.utils.data.DataLoader(deploy_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, deploy_loader
