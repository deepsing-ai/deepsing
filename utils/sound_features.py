import librosa
import librosa.display
import numpy as np
from utils.config import config


def extract_features_from_file(path, n_mfcc=40, annotation_period=500, hop_length=512, sr=21504):
    """
    Extracts MFCC, Tempograpm and Chroma features from an audio file
    :param path:
    :param n_mfcc:
    :param annotation_period:
    :param hop_length:
    :param sr:
    :return:
    """
    y, sr = librosa.load(path, sr=sr)

    mfcc_features = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, hop_length=hop_length)
    tempo_features = librosa.feature.tempogram(y, sr, hop_length=hop_length)
    chroma_features = librosa.feature.chroma_cens(y, sr, hop_length=hop_length)
    spec_features = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    # spec_features = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
    # Keep the upper 96 mels
    # spec_features = spec_features[1:32:, :]
    # spec_features = np.gradient(np.float64(spec_features))

    # 40 - 384 - 12
    features = np.concatenate([mfcc_features, tempo_features, chroma_features])

    # Discard the last window if not even
    if features.shape[1] % 2:
        features = features[:, :-1]
        spec_features = spec_features[:, :-1]
    features = features.T
    spec_features = spec_features.T

    # Calculate the duration for each extracted feature
    features_sr_msec = 1000 * (hop_length / sr)

    # Calculate how many features are needed for one annotation period
    feats_per_annotation_period = int(annotation_period / features_sr_msec)

    # Clip to the required length
    n_feats = int(features.shape[0] / feats_per_annotation_period) * feats_per_annotation_period
    features = features[:n_feats]
    spec_features = spec_features[:n_feats]


    features = features.reshape(-1, feats_per_annotation_period * features.shape[1])
    features = np.float16(features)


    spec_features = spec_features.reshape(-1, feats_per_annotation_period, spec_features.shape[1])
    spec_features = np.mean(np.mean(np.float16(spec_features), axis=-1), axis=-1)
    #spec_features = np.gradient(spec_features)
    # spec_features = spec_features / np.max(spec_features)
    # spec_features = np.abs(spec_features)

    return features, spec_features
