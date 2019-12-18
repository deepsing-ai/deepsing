import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNetV2
import pickle
from utils.config import config
from os.path import join

class SoundMLP(nn.Module):
    def __init__(self, input_size=9156):
        super(SoundMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 2)

        self.mean = 0
        self.std = 1

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        out = self.fc1(x)
        out = torch.sigmoid(out)

        out = self.fc2(out)
        return out

    def get_np_features(self, x):
        x = self.forward(x)
        x = x.cpu().detach().numpy()
        x = (x - self.mean) / self.std

        return x

    def load_neural_space_statistics(self, path=join(config['basedir'], "models/space_statistics.pickle")):
        with open(path, "rb") as f:
            self.mean = pickle.load(f)
            self.std = pickle.load(f)

    def load_neural_model(self, path=join(config['basedir'],"models/mlp.model")):
        self.load_state_dict(torch.load(path, map_location=torch.device(config['device'])))

    def save_neural_space_statistics(self, path=join(config['basedir'], "models/space_statistics.pickle")):
        with open(path, "wb") as f:
            pickle.dump(self.mean, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.std, f, protocol=pickle.HIGHEST_PROTOCOL)


class NeuralTranslator(nn.Module):

    def __init__(self, input_size=2):
        super(NeuralTranslator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3_class = nn.Linear(256, 1000)
        self.fc3_noise = nn.Linear(256, 128)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        selected_class = self.fc3_class(out)
        selected_noise = self.fc3_noise(out)
        return (selected_class, selected_noise)


def get_pretrained_mobile_net(pretrained=True):
    if pretrained:
        pretrained_net = mobilenet_v2(pretrained=True)

    net = MobileNetV2(num_classes=2)
    if pretrained:
        state_dict = pretrained_net.state_dict()
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']
        net.load_state_dict(state_dict, strict=False)

    return net
