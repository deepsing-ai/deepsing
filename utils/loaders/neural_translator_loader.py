from PIL import Image
from utils.models import get_pretrained_mobile_net
from torch.utils.data import Dataset
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample)
import pickle
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from utils.loaders.neural_translator_utils import get_gan_space_view, get_gan_space_view_from_ids
from utils.config import config
from os.path import join

class GANOnlineHelperLoader(Dataset):

    def __init__(self, image_sentiment_model=join(config['basedir'], "models/image_sentiment.model"), noise=0.3, in_batch=4):
        # Load GAN
        self.gan_model = BigGAN.from_pretrained('biggan-deep-512')
        self.gan_model.to(config['device'])

        # Load image sentiment analysis model
        self.sentiment_model = get_pretrained_mobile_net()
        self.sentiment_model.to(config['device'])
        self.sentiment_model.load_state_dict(torch.load(image_sentiment_model))

        self.in_batch = in_batch
        self.noise = noise
        self.n_iters = int(1000 / in_batch)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_image = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize, ])

    def __len__(self):
        return self.n_iters

    def __getitem__(self, idx):
        if idx > self.n_iters:
            raise StopIteration

        # Sample the space
        idx = np.random.randint(0, 1000, self.in_batch)
        class_vectors = one_hot_from_int(idx, batch_size=self.in_batch)
        noise_vectors = truncated_noise_sample(truncation=self.noise, batch_size=self.in_batch)
        class_vectors = torch.tensor(class_vectors).to(config['device'])
        noise_vectors = torch.tensor(noise_vectors).to(config['device'])

        with torch.no_grad():
            output = self.gan_model(noise_vectors, class_vectors, self.noise)

        # Convert to PIL Image
        output = output.detach().cpu().numpy()
        output = np.uint8(np.clip(((output + 1) / 2.0) * 256, 0, 255))
        output = output.transpose((0, 2, 3, 1))
        images = []

        # Pre-process each image to feed them into image sentiment analysis model
        for i in range(output.shape[0]):
            cur_img = Image.fromarray(output[i])
            cur_img = self.transform_image(cur_img)
            images.append(cur_img)
        images = torch.stack(images).to(config['device'])

        # Feed-forward image sentiment analysis
        sentiment = self.sentiment_model(images)

        class_vectors = class_vectors.cpu().detach().numpy()
        noise_vectors = noise_vectors.cpu().detach().numpy()
        sentiment = sentiment.cpu().detach().numpy()

        return class_vectors, noise_vectors, sentiment


def create_collection(epochs=1, dataset_path=join(config['basedir'],'data/neural_translation_dataset.pickle')):
    """
    Samples the GAN space and creates a dataset for training the translator
    Note that the Online Translator can be also directly used for generating the dataset online.
    However, caching the dataset allows for more easily conduncting experiments (since generating images using the GAN
    is not cheap)

    :param epochs: Number of generation epochs (in each epoch 1000 images (number of classes in ILSVRC) are generated)
    :param dataset_path: path to save the generated dataset
    :return:
    """
    s = GANOnlineHelperLoader(in_batch=5)
    class_vecs, noise_vecs, sent_vecs = [], [], []

    for i in range(epochs):
        print("Epoch: ", i)
        for cur_class, cur_noise, cur_sentiment in tqdm(s):
            class_vecs.append(cur_class)
            noise_vecs.append(cur_noise)
            sent_vecs.append(cur_sentiment)
    class_vecs = np.concatenate(np.float16(class_vecs))
    noise_vecs = np.concatenate(np.float16(noise_vecs))
    sent_vecs = np.concatenate(np.float16(sent_vecs))

    with open(dataset_path, "wb") as f:
        pickle.dump(class_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(noise_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sent_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)


class NeuralTranslatorLoader(Dataset):

    def __init__(self, dataset_path=join(config['basedir'], 'data/neural_translation_dataset.pickle'),
                 train=False, deploy=True, n_clusters=10,
                 seed=1, n_sub_classes=5, ids=None):

        if ids is not None:
            class_vecs, noise_vecs, sent_vecs = get_gan_space_view_from_ids(dataset_path=dataset_path, selected_ids=ids,
                                                                            n_sub_classes=n_sub_classes)
        else:
            class_vecs, noise_vecs, sent_vecs = get_gan_space_view(dataset_path=dataset_path, n_sub_classes=n_sub_classes,
                                                               n_clusters=n_clusters, seed=seed)
        print("Selected classes: ", np.unique(class_vecs.argmax(1)))

        self.class_vecs = class_vecs
        self.noise_vecs = noise_vecs
        self.sent_vecs = sent_vecs

        # Use the 70\% of the dataset of training
        thres = int(0.7 * len(self.class_vecs))
        print("Training samples = ", len(self.class_vecs))

        if train and not deploy:
            self.class_vecs = self.class_vecs[:thres]
            self.noise_vecs = self.noise_vecs[:thres]
            self.sent_vecs = self.sent_vecs[:thres]
        elif not train and not deploy:
            self.class_vecs = self.class_vecs[thres:]
            self.noise_vecs = self.noise_vecs[thres:]
            self.sent_vecs = self.sent_vecs[thres:]

    def __len__(self):
        return len(self.class_vecs)

    def __getitem__(self, idx):
        # The valence-arousal is the input to the model
        data = torch.tensor(np.float32(self.sent_vecs[idx]))

        # The corresponding GAN input that leads to the sentiment is the target
        target_class = torch.tensor(np.float32(self.class_vecs[idx]))
        target_noise = torch.tensor(np.float32(self.noise_vecs[idx]))

        return data, (target_class, target_noise,)


def get_train_loaders(batch_size=32, n_clusters=5, seed=1, n_sub_classes=5, ids=None):
    train_dataset = NeuralTranslatorLoader(train=True, deploy=False, n_clusters=n_clusters, seed=seed,
                                           n_sub_classes=n_sub_classes, ids=ids)
    val_dataset = NeuralTranslatorLoader(train=False, deploy=False, n_clusters=n_clusters, seed=seed,
                                         n_sub_classes=n_sub_classes, ids=ids)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_deploy_loaders(batch_size=32, n_clusters=5, seed=1, n_sub_classes=5, ids=None):
    train_dataset = NeuralTranslatorLoader(train=True, deploy=True, n_clusters=n_clusters, seed=seed,
                                           n_sub_classes=n_sub_classes, ids=ids)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
