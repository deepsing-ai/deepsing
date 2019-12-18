import pickle
from sklearn.cluster import KMeans
import collections
import numpy as np


def select_class_per_cluster(cluster_idx, class_ids, n_clusters):
    """
    Samples a class per cluster, according to the number of samples for the corresponding class
    :param cluster_idx:
    :param class_ids:
    :param n_clusters:
    :return:
    """
    selected_classes = []
    for i in range(n_clusters):
        cur_idx = cluster_idx == i
        cur_classes = class_ids[cur_idx]
        histogram = collections.Counter(cur_classes.flatten())

        probabilities = []
        classes = []
        for x in histogram:
            if x > 398:
                classes.append(x)
                probabilities.append(histogram[x])
        probabilities = np.asarray(probabilities) / np.sum(probabilities)

        c = np.random.choice(classes, p=probabilities)
        selected_classes.append(c)
    selected_classes = np.asarray(selected_classes)
    return selected_classes


def smoothen_with_sub_classes(class_vecs, noise_vecs, sent_vecs, sub_classes=2):
    """
    Performs clustering to detect sub-classes and then replaces the sentiment
    with the mean sentiment of each sub-class. This allows for smoothening the training
    process (both for estimating the class and the noise for the classes)
    :param class_vecs:
    :param noise_vecs:
    :param sent_vecs:
    :param sub_classes:
    :return:
    """
    class_ids = class_vecs.argmax(1)
    print("# subclasses = ", sub_classes)
    for i in np.unique(class_ids):
        idx = class_ids == i

        class_sentiments = sent_vecs[idx]
        noise = noise_vecs[idx]

        # Cluster the sentiment space
        kmeans = KMeans(n_clusters=sub_classes)
        cluster_idx = kmeans.fit_predict(class_sentiments)
        for j in range(sub_classes):
            c_idx = cluster_idx == j
            class_sentiments[c_idx] = np.mean(class_sentiments[c_idx], axis=0)
            noise[c_idx] = np.mean(noise[c_idx], axis=0)
        sent_vecs[idx] = class_sentiments
        noise_vecs[idx] = noise
    return class_vecs, noise_vecs, sent_vecs



def get_gan_space_view_from_ids(dataset_path='../../data/neural_translation_dataset.pickle', selected_ids=None, n_sub_classes=5):

    # Load dataset
    with open(dataset_path, "rb") as f:
        class_vecs = pickle.load(f)
        noise_vecs = pickle.load(f)
        sent_vecs = pickle.load(f)
    class_ids = class_vecs.argmax(1)

    # Normalize sentiments similarly to the normalization performed for the audio model
    sent_vecs = (sent_vecs - np.mean(sent_vecs, axis=0)) / np.std(sent_vecs, axis=0)

    idx = False
    for cur_id in selected_ids:
        idx = np.logical_or(idx, class_ids == cur_id)

    class_vecs = class_vecs[idx]
    noise_vecs = noise_vecs[idx]
    sent_vecs = sent_vecs[idx]

    class_vecs, noise_vecs, sent_vecs = smoothen_with_sub_classes(class_vecs, noise_vecs, sent_vecs,
                                                                  sub_classes=n_sub_classes)

    return class_vecs, noise_vecs, sent_vecs



def get_gan_space_view(dataset_path='data/neural_translation_dataset.pickle', n_clusters=10, seed=1,
                       n_sub_classes=5):
    """
    Gets a "view" of the GAN generation space
    Note that sentiment cannot be used to uniquely identify the classes support by the GAN, with multiple classes
    leading more or less to the same sentiment. To this end, here we cluster the sentiment space and we randomly select
    one class per cluster. We also further smooth the sentiment space by detecting subclasses (this was neccersary to improve
    the stability of the model).
    :param dataset_path:
    :param n_clusters:
    :param seed:
    :param n_sub_classes:
    :return:
    """
    np.random.seed(seed)

    # Load dataset
    with open(dataset_path, "rb") as f:
        class_vecs = pickle.load(f)
        noise_vecs = pickle.load(f)
        sent_vecs = pickle.load(f)
    class_ids = class_vecs.argmax(1)

    # Normalize sentiments similarly to the normalization performed for the audio model
    sent_vecs = (sent_vecs - np.mean(sent_vecs, axis=0)) / np.std(sent_vecs, axis=0)

    # Cluster the sentiment space
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_idx = kmeans.fit_predict(sent_vecs)
    selected_classes = select_class_per_cluster(cluster_idx, class_ids, n_clusters)

    idx = False
    for cur_id in selected_classes:
        idx = np.logical_or(idx, class_ids == cur_id)

    class_vecs = class_vecs[idx]
    noise_vecs = noise_vecs[idx]
    sent_vecs = sent_vecs[idx]

    class_vecs, noise_vecs, sent_vecs = smoothen_with_sub_classes(class_vecs, noise_vecs, sent_vecs,
                                                                  sub_classes=n_sub_classes)

    return class_vecs, noise_vecs, sent_vecs
