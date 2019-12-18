from torch.utils.data import Dataset
import pickle
import torch
import numpy as np


class SentimentDataset(Dataset):

    def __init__(self, dataset_path="data/dataset.pickle", train=False, deploy=False, window=50):

        with open(dataset_path, "rb") as f:
            features = pickle.load(f)
            labels = pickle.load(f)

        self.window = window

        self.features = features
        self.labels = labels

        if train and not deploy:
            self.features = features[:1500]
            self.labels = labels[:1500]
        elif not train and not deploy:
            self.features = features[1500:]
            self.labels = labels[1500:]
        else:
            print("Using the whole dataset!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, labels = self.features[idx], self.labels[idx]

        # Sub-sample if needed
        if data.shape[0] > self.window:
            idx = np.random.permutation(data.shape[0])[:self.window]
            data = data[idx, :]
            labels = labels[idx, :]

        data = torch.tensor(np.float32(data))
        labels = torch.tensor(labels)
        return data, labels


def get_train_loaders(batch_size=128, dataset_path="data/dataset.pickle"):
    train_dataset = SentimentDataset(train=True, deploy=False, dataset_path=dataset_path)
    val_dataset = SentimentDataset(train=False, deploy=False, dataset_path=dataset_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_deploy_loaders(batch_size=128, dataset_path="data/dataset.pickle"):
    train_dataset = SentimentDataset(train=True, deploy=True, dataset_path=dataset_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
