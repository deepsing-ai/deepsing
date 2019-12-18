import torch
from utils.models import NeuralTranslator
from utils.loaders.neural_translator_loader import create_collection, get_train_loaders, get_deploy_loaders
from utils.train_utils import train_model, eval_network
from tqdm import tqdm
from utils.config import config


def create_dataset(epochs=100):
    """
    Samples the GAN space and caches the corresponding sentiments
    (You can also use the online generator)
    :param epochs:
    :return:
    """
    create_collection(epochs=epochs)


def train_custom_id(imagenet_ids, i):
    train_loader = get_deploy_loaders(ids=imagenet_ids[i], n_sub_classes=10)

    net = NeuralTranslator()
    net.to(config['device'])

    train_model(net, train_loader, epochs=200, lr=0.001, train_type='translator')
    eval_network(net, train_loader, train_type='translator')
    torch.save(net.state_dict(), "models/neural_translator_custom_" + str(i) + ".model")

    net = NeuralTranslator()
    net.to(config['device'])
    net.load_state_dict(torch.load("models/neural_translator_custom_" + str(i) + ".model"))
    eval_network(net, train_loader, train_type='translator')


def train_deploy():
    imagenet_ids = [[970, 497, 850, 470, 698],
                    [743, 437, 471, 475, 583, 483, 552, 402, 542],
                    [476, 463, 995, 691,  764, 913, 812, 992, 845],
                    [833, 403, 100,  724, 744, 638, 639, 847]
                    ]

    train_custom_id(imagenet_ids, 3)

    # for i in range(len(imagenet_ids)):
    #     train_custom_id(imagenet_ids, i)


if __name__ == '__main__':
    train_deploy()
