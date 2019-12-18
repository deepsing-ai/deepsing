import torch
from tqdm import tqdm
from utils.video_utils import torch_to_cv2
from utils.config import config


def feedforward_gan(model, class_vectors, noise_vectors, batch_size, truncation):
    """
    Feedd-fowards the GAN and creates a collection of images
    :param model:
    :param class_vectors:
    :param noise_vectors:
    :param batch_size:
    :param class_ids:
    :param truncation:
    :return:
    """

    images = []
    n_batches = int(len(class_vectors) / batch_size)

    print("Generating GAN content...")
    for i in tqdm(range(n_batches)):
        cur_noise = torch.from_numpy(noise_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])
        cur_class = torch.from_numpy(class_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())

    if n_batches * batch_size < len(class_vectors):
        cur_noise = torch.from_numpy(noise_vectors[n_batches * batch_size:]).to(config['device'])
        cur_class = torch.from_numpy(class_vectors[n_batches * batch_size:]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())

    images = torch_to_cv2(images)
    return images
