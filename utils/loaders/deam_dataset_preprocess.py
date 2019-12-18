from os.path import join
import csv
import numpy as np
from tqdm import tqdm
from utils.sound_features import extract_features_from_file
import pickle
import multiprocessing

anotations_path = '/home/nick/Data/Datasets/audio/annotations/annotations averaged per song/dynamic (per second annotations)'
songs_path = '/home/nick/Data/Datasets/audio/audio'


def load_annotations(path):
    """
    Loads the DEAM annotations (per audio file)
    :param path:
    :return:
    """
    # Load arousal values
    with open(join(path, "arousal.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        x = list(reader)

    header = x[0]
    # Get the annotation time-stamps in ms
    timestamps = [int(x[7:-2]) for x in header[1:]]

    # Get the labels
    data = x[1:]
    arousal_labels = [np.float32(x[1:]) for x in data]

    # Load valence values
    with open(join(path, "valence.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        x = list(reader)

    # Get the labels
    data = x[1:]
    song_id = [np.int64(x[0]) for x in data]
    valence_labels = [np.float32(x[1:]) for x in data]

    return song_id, timestamps, arousal_labels, valence_labels


def process_song(current_path):
    """
    Helper function for parallelizing features extraction
    :param current_path:
    :return:
    """
    song_features = extract_features_from_file(current_path)
    return song_features


def preprocess_dataset(n_threads=6):
    song_id, timestamps, arousal_labels, valence_labels = load_annotations(anotations_path)
    skip_features = int(int(timestamps[0] / 1000) / 0.5) - 1

    song_paths = [join(songs_path, str(x) + '.mp3') for x in song_id]
    print("Number of songs in path = ", len(song_paths))

    pool = multiprocessing.Pool(n_threads)
    features = pool.map(process_song, song_paths)
    labels = []

    for i in tqdm(range(len(song_id))):
        song_features = features[i]
        # Skip the first 15sec according to the annotations of the dataset
        song_features = song_features[skip_features:]
        n_max = min(len(song_features), min(len(arousal_labels[i]), len(valence_labels[i])))

        # Clip according to the available annotations
        song_features = song_features[:n_max]
        song_labels = np.float32([arousal_labels[i][:n_max], valence_labels[i][:n_max]]).T

        features[i] = song_features
        labels.append(song_labels)

    with open("dataset.pickle", "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    preprocess_dataset()
