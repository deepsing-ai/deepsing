from pytorch_pretrained_biggan import (truncated_noise_sample)
from utils.models import NeuralTranslator
import torch
from utils.config import config
from os.path import join

class NeuralSentimentTranslator:

    def __init__(self, translator_path=join(config['basedir'], "models/neural_translator_0.model"),
                 class_path=join(config['basedir'], "models/imagenet_original.txt"), temperature=0.05, noise_scaler=1):

        # Load the imagenet classes
        with open(class_path, 'r') as inf:
            self.imagenet_classes = eval(inf.read())

        self.temperature = temperature
        self.noise_scaler = noise_scaler
        self.net = NeuralTranslator()
        self.net.to(config['device'])
        self.net.load_state_dict(torch.load(translator_path))

    def translate_sentiment(self, audio_sentiment, noise):
        song_words = []

        audio_sentiment = torch.tensor(audio_sentiment).to(config['device'])
        class_vectors, noise_vectors = self.net(audio_sentiment)
        class_vectors = torch.softmax(class_vectors / self.temperature, dim=1)
        cur_noise = torch.tensor(truncated_noise_sample(batch_size=len(noise_vectors), truncation=noise)).to(config['device'])
        noise_vectors = noise_vectors * self.noise_scaler + cur_noise

        # Get the names of the imagenet classes
        class_ids = class_vectors.argmax(1).cpu().detach().numpy()
        for i in class_ids:
            song_words.append(self.imagenet_classes[i])

        class_vectors, noise_vectors = class_vectors.cpu().detach().numpy(), noise_vectors.cpu().detach().numpy()
        return class_vectors, noise_vectors, song_words
