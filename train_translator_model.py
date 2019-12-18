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


def train_and_evaluate():
    net = NeuralTranslator()
    net.to(config['device'])
    train_loader, test_loader = get_train_loaders(n_clusters=10, seed=1, n_sub_classes=10)

    train_model(net, train_loader, epochs=200, lr=0.001, train_type='translator')
    eval_network(net, train_loader, train_type='translator')
    eval_network(net, test_loader, train_type='translator')




def train_deploy(seed=1):
    train_loader = get_deploy_loaders(n_clusters=10, seed=seed, n_sub_classes=10)

    net = NeuralTranslator()
    net.to(config['device'])

    train_model(net, train_loader, epochs=200, lr=0.001, train_type='translator')
    eval_network(net, train_loader, train_type='translator')
    torch.save(net.state_dict(), "models/neural_translator_" + str(seed) + ".model")

    net = NeuralTranslator()
    net.to(config['device'])
    net.load_state_dict(torch.load("models/neural_translator_" + str(seed) + ".model"))
    eval_network(net, train_loader, train_type='translator')


if __name__ == '__main__':
    # create_dataset()

    # train_and_evaluate()

    # Create 5 views
    for i in tqdm(range(20)):
        train_deploy(seed=i)
