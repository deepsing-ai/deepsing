from sklearn.neighbors import NearestNeighbors
from pytorch_pretrained_biggan import (one_hot_from_int, truncated_noise_sample)
import csv
import numpy as np
import pickle
from utils.config import config
from os.path import join

class TextualSentimentTranslator:

    def __init__(self, class_dictionary_path=join(config['basedir'], "models/imagenet.txt"), k_neighbors=1, temperature=1):
        self.temperature = temperature
        ids, features, words = load_sentiment_embeddings(class_dictionary_path)
        self.class_ids = ids
        self.features = features
        self.words = words
        self.database = NearestNeighbors(n_neighbors=k_neighbors)
        self.database.fit(features, range(len(features)))

    def translate_sentiment(self, audio_sentiment, noise=0.1):

        # Step 1: Find the closest word match to the audio sentiment
        dists, idx = self.database.kneighbors(audio_sentiment)
        song_words = [self.words[cur_index[0]] for cur_index in idx]
        song_words = [x.split(',')[0].lower() for x in song_words]
        word_ids = [self.class_ids[cur_index] for cur_index in idx]

        class_vectors = []
        noise_vectors = []

        # Calculate the weighted nearest class neighbors
        for i in range(len(word_ids)):
            cur_class_vector = one_hot_from_int(word_ids[i], batch_size=len(word_ids[i]))

            # If k = 1
            if len(word_ids[i]) == 1:
                cur_class_vector = cur_class_vector[0]
            else:
                similarities = np.exp(-dists[i] / self.temperature)
                similarities = similarities / np.sum(similarities)
                similarities = similarities.reshape((-1, 1))
                cur_class_vector = np.sum(cur_class_vector * similarities, axis=0)

            class_vectors.append(cur_class_vector)
            noise_vectors.append(truncated_noise_sample(truncation=noise, batch_size=1)[0])

        class_vectors = np.float32(class_vectors)
        noise_vectors = np.float32(noise_vectors)

        return class_vectors, noise_vectors, song_words


def create_affective_dictionary(lexicon_path=join(config['basedir'], "models/words.csv"),
                                class_path=join(config['basedir'],'models/imagenet.txt'), standarize=True):
    with open(lexicon_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        x = list(reader)

    print("Valence source: ", x[0][2])
    print("Arousal source: ", x[0][5])

    # Skip the header and load the words
    x = x[1:]
    words = []
    features = []

    for i in range(len(x)):
        word, valence, arousal = x[i][1], x[i][2], x[i][5]
        words.append(word)
        features.append((float(arousal), float(valence)))
    features = np.asarray(features)

    # Convert to dictionary
    word_dict = {}
    for word, vec in zip(words, features):
        word_dict[word] = vec

    # Load the imagenet classes
    with open(class_path, 'r') as inf:
        imagenet = eval(inf.read())

    imagenet_id = []
    imagenet_features = []
    imagenet_words = []

    for id in imagenet:
        desc = imagenet[id]
        desc = desc.replace(',', ' ').lower().split(' ')

        cur_vec = np.zeros((2,))
        count = 0
        for word in desc:
            if word in word_dict:
                cur_vec += word_dict[word]
                count += 1

        if count > 0:
            cur_vec = cur_vec / count
            imagenet_id.append(id)
            imagenet_features.append(cur_vec)
            imagenet_words.append(imagenet[id])

    imagenet_id = np.asarray(imagenet_id)
    imagenet_features = np.asarray(imagenet_features)

    if standarize:
        mean = np.mean(imagenet_features, axis=0)
        std = np.std(imagenet_features, axis=0)

        imagenet_features = (imagenet_features - mean) / std

    # with open(output_path, "wb") as f:
    #     pickle.dump(imagenet_id, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(imagenet_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(imagenet_words, f, protocol=pickle.HIGHEST_PROTOCOL)

    return imagenet_id, imagenet_features, imagenet_words


def load_sentiment_embeddings(path, lexicon_path=join(config['basedir'], "models/words.csv")):
    ids, features, words = create_affective_dictionary(lexicon_path=lexicon_path, class_path=path,
                                                       standarize=True)

    #
    # with open(path, "rb") as f:
    #     ids = pickle.load(f)
    #     features = pickle.load(f)
    #     words = pickle.load(f)
    return ids, features, words
