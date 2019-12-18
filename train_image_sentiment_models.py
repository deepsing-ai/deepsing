from utils.models import get_pretrained_mobile_net
from utils.loaders.oasis_loaders import get_oasis_dataset_loaders
from utils.train_utils import train_model, eval_network
import torch
from utils.config import config

def train_and_evaluate():
    net = get_pretrained_mobile_net(pretrained=True)
    net.to(config['device'])
    train_loader, test_loader, deploy_loader = get_oasis_dataset_loaders()

    for i in range(10):
        print(i*5)
        train_model(net, train_loader, epochs=5, lr=0.0001, train_type='image')
        eval_network(net, test_loader, train_type='image')


def train_deploy():
    train_loader, test_loader, deploy_loader = get_oasis_dataset_loaders()

    net = get_pretrained_mobile_net(pretrained=True)
    net.to(config['device'])

    train_model(net, deploy_loader, epochs=50, lr=0.0001, train_type='image')
    eval_network(net, test_loader, train_type='image')
    train_model(net, deploy_loader, epochs=10, lr=0.00001, train_type='image')
    eval_network(net, test_loader, train_type='image')
    torch.save(net.state_dict(), "models/image_sentiment.model")

    model = get_pretrained_mobile_net(pretrained=True)
    model.to(config['device'])
    model.load_state_dict(torch.load("models/image_sentiment.model"))
    eval_network(model, test_loader, train_type='image')

if __name__ == '__main__':
    train_deploy()


