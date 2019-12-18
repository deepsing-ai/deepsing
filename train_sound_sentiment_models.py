from utils.models import SoundMLP
from utils.train_utils import train_model, eval_network, get_features
from utils.loaders.deam_loader import get_train_loaders, get_deploy_loaders
import numpy as np
import torch
from utils.config import config


def train_and_evaluate():
    # Just for measure the performance of the sentiment analysis model
    net = SoundMLP()
    net.to(config['device'])
    train_loader, test_loader = get_train_loaders()
    train_model(net, train_loader, epochs=30, lr=0.0001)
    eval_network(net, test_loader)


def train_deploy():
    train_loader = get_deploy_loaders()

    net = SoundMLP()
    net.to(config['device'])

    train_model(net, train_loader, epochs=30, lr=0.0001)
    eval_network(net, train_loader)
    train_model(net, train_loader, epochs=20, lr=0.00001)
    eval_network(net, train_loader)

    torch.save(net.state_dict(), "models/mlp.model")

    model = SoundMLP()
    model.to(config['device'])
    model.load_state_dict(torch.load("models/mlp.model"))

    eval_network(model, train_loader)

def calculate_scaling_model_scaling():
    train_loader = get_deploy_loaders()

    model = SoundMLP()
    model.to(config['device'])
    model.load_neural_model("models/mlp.model")

    features = get_features(model, train_loader)
    features = features.reshape((-1, 2))

    model.mean = np.mean(features, axis=0)
    model.std = np.std(features, axis=0)

    model.save_neural_space_statistics("models/space_statistics.pickle")

    print("Mean = ", model.mean)
    print("Std = ", model.std)

def validate_saved_model():
    train_loader = get_deploy_loaders()

    model = SoundMLP()
    model.to(config['device'])
    model.load_neural_model("models/mlp.model")
    model.load_neural_space_statistics("models/space_statistics.pickle")

    eval_network(model, train_loader)

    print("Mean = ", model.mean)
    print("Std = ", model.std)


if __name__ == '__main__':
    # Step 0: Train the model
    train_deploy()

    # Step 1: Standarize the feature space
    calculate_scaling_model_scaling()

    # Ensure that we saved the model correctly
    validate_saved_model()
