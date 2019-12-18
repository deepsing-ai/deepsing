#!/usr/bin/env python3

# Deploy hack - I am too lazy to do things in the right way :P
# Modify this to link to the base opendr
import sys
sys.path.append('.')

from os.path import join
from utils.deep_sing import DeepSing
from stylization_utils.stylizer import Stylizer
from utils.textual_translator import TextualSentimentTranslator
from utils.sound_sentiment import SoundSentimentExtractor
from utils.neural_translator import NeuralSentimentTranslator
from utils.config import config
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='deepsing: music-to-image translator')

parser.add_argument('input', type=str, help='song to translate')
parser.add_argument('output', type=str, help='path to save the song (please do not provide any suffix)')
parser.add_argument('--translator', type=str, help='translator to use',
                    default="textual")
parser.add_argument('--path', type=str, help='path to translator models',
                    default=join(config['basedir'], 'models/neural_translator_'))
parser.add_argument('--nid', type=int, help='id of the translator model, if -1 a random model is chosen',
                    default=-1)
parser.add_argument('--subsample', type=int, help='number of 500msec intervals to stay on each class',
                    default=10)
parser.add_argument('--dynamic', help='generate one frame per each 500msec interval', action='store_true')
parser.add_argument('--dictionary', type=str,
                    help='json dictionary with imagenet classes (only used with textual translators)',
                    default=join(config['basedir'], 'models/imagenet_clean.txt'))
parser.add_argument('--noinfo', help='disables the generation of on-video info', action='store_true', default=True)
parser.add_argument('--raw', help='perform song-basis normalization', action='store_true')
parser.add_argument('--nostylizer', help='disables the stylization', action='store_true')
parser.add_argument('--stylizer-path', type=str,
                    help='path with stylization images (positive, negative, neutral)',
                    default=join(config['basedir'], 'resources/sentiment_images'))
parser.add_argument('--stylization-factor', type=float,
                    help='stylization factor (higher values leads to more stylization)',
                    default=0.1)
parser.add_argument('--stylization-threshold', type=float,
                    help='threshold for switching between classes used for stylization',
                    default=0.5)

args = parser.parse_args()

if args.translator == 'neural':

    if args.nid < 0:
        id = np.random.randint(0, 23, 1)[0]
        print("Translator selected: ", id)
    else:
        id = args.nid

    args.path = args.path + str(id) + '.model'
    translator = NeuralSentimentTranslator(translator_path=args.path)
elif args.translator == 'textual':
    translator = TextualSentimentTranslator(k_neighbors=1, class_dictionary_path=args.dictionary)
else:
    print("Unknown translator! Supported translators: 'neural', 'textual'")

sound_sentiment = SoundSentimentExtractor()

if not args.nostylizer:
    stylizer = Stylizer(stylization_factor=args.stylization_factor, neutral_threshold=args.stylization_threshold)
    stylizer.load_images(args.stylizer_path)
else:
    stylizer = None

deep_sing = DeepSing(sound_sentiment, translator, stylizer_model=stylizer)
deep_sing.draw_song(args.input, args.output, smoothing_window=1, noise=0.1, subsample=args.subsample,
                    normalize=not args.raw, debug=not args.noinfo, use_transition=args.dynamic)
