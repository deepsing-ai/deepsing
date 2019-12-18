from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from stylization_utils.model import MultiLevelAE
from os.path import join
import numpy as np
from tqdm import tqdm
import cv2
from utils.config import config
from os.path import join

class Stylizer:

    def __init__(self, neutral_image=None, positive_image=None, negative_image=None, neutral_threshold=1,
                 stylization_model_path=join(config['basedir'], 'models/stylization_model'),
                 stylization_factor=0.2):

        self.model = MultiLevelAE(stylization_model_path)
        self.model.to(config['device'])

        # Save the reference images
        self.neutral_image = neutral_image
        self.positive_image = positive_image
        self.negative_image = negative_image

        # Threshold for considering an image positive or negative
        self.neutral_threshold = neutral_threshold
        self.stylization = stylization_factor

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.trans = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    def load_images(self, images_path=join(config['basedir'], 'resources/sentiment_images')):

        self.neutral_image = Image.open(join(images_path, 'neutral.jpg'))
        self.positive_image = Image.open(join(images_path, 'positive.jpg'))
        self.negative_image = Image.open(join(images_path, 'negative.jpg'))

    def _wct_stylization(self, content_image, style_image):
        c_tensor = self.trans(content_image).unsqueeze(0).to(config['device'])
        s_tensor = self.trans(style_image).unsqueeze(0).to(config['device'])
        with torch.no_grad():
            out = self.model(c_tensor, s_tensor, self.stylization)

        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(config['device'])
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(config['device'])
        out = torch.clamp(out * std + mean, 0, 1)

        return out

    def sentiment_stylization(self, content_image, sentiment_vector):
        sentiment = None
        if torch.mean(sentiment_vector) > self.neutral_threshold:
            style_image = self.positive_image
            sentiment = 'positive'
        elif torch.mean(sentiment_vector) < -self.neutral_threshold:
            style_image = self.negative_image
            sentiment = 'negative'
        else:
            style_image = self.neutral_image
            sentiment = 'neutral'

        return self._wct_stylization(content_image, style_image), sentiment


def stylize_frames(images, stylizer, sentiments):
    stylizer_sentiments = []
    print("Stylizing images...")
    for i in tqdm(range(len(images))):
        cur_img = images[i]
        cur_sentiment = torch.tensor(sentiments[i])
        # BGR TO RGB
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        cur_img = Image.fromarray(cur_img)
        cur_img, cur_sentiment = stylizer.sentiment_stylization(cur_img, cur_sentiment)
        stylizer_sentiments.append(cur_sentiment)
        cur_img = cur_img.detach().cpu().numpy()[0]
        # RGB TO BGR
        cur_img = np.uint8(np.clip(cur_img * 256, 0, 255))
        cur_img = cur_img.transpose((1, 2, 0))[:, :, ::-1].copy()
        images[i] = cur_img
    return stylizer_sentiments

# def main():
#
#     c = Image.open('/home/nick/Data/Workspace/Forum/deep_sing/resources/in1.jpg')
#     print(np.asarray(c))
#     neutral_image = Image.open('/home/nick/Data/Workspace/Forum/deep_sing/resources/sentiment_images/neutral.jpg')
#     positive_image = Image.open('/home/nick/Data/Workspace/Forum/deep_sing/resources/sentiment_images/positive.jpg')
#     negative_image = Image.open('/home/nick/Data/Workspace/Forum/deep_sing/resources/sentiment_images/negative.jpg')
#
#
#     stylizer = Stylizer(neutral_image=neutral_image, positive_image=positive_image, negative_image=negative_image)
#     out = stylizer.sentiment_stylization(c, torch.tensor([-2.0, -2.0]))
#     save_image(out, 'test1.jpg', nrow=1)
#     out = stylizer.sentiment_stylization(c, torch.tensor([0.0, 0.0]))
#     save_image(out, 'test2.jpg', nrow=1)
#     out = stylizer.sentiment_stylization(c, torch.tensor([2.0, 2.0]))
#     save_image(out, 'test3.jpg', nrow=1)
#
#
# if __name__ == '__main__':
#     main()
