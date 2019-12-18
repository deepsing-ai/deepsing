# deepsing: Generating Sentiment-aware Visual Stories using Cross-modal Music Translation

**deepsing** is a deep learning method for performing attributed-based music-to-image translation. The proposed method is capable of synthesizing visual stories according to the sentiment expressed by songs. The generated images aim to induce the same feelings to the viewers, as the original song does, reinforcing the primary aim of music, i.e., communicating feelings. deepsing employs a **trainable cross-modal translation method** to generate visual stories, as shown below:

![alt text](https://raw.githubusercontent.com/deepsing-ai/deepsing/master/pictures/pipeline.png "deepsing pipeline")

The process of generating a visual story is also illustrated bellow

![alt text](https://raw.githubusercontent.com/deepsing-ai/deepsing/master/pictures/example.png "example of generating a visual story")


To use deepsing please install PyTorch, as well as the rest of the required dependecies:
```
pip install pytorch-pretrained-biggan librosa scipy 
sudo apt-get install ffmpeg
```

Also, be sure to download the [pre-trained models](https://drive.google.com/file/d/1r72i-F9YaJ7tKil0SJkW35tHIdi4Ci3r/view?usp=sharing) and unzip them into the [models](https://github.com/deepsing-ai/deepsing/tree/master/models) folder.

You can directly run *deepsing* on an audio file (e.g., sample.mp3) as
```
./deepsing.py temp/sample.mp3 temp/output
```
Please run *deepsing* from the root folder (or set the appropriate paths respectively).

Note that by default we use a *dictionary*-based translator. To use the proposed neural translation approach you can set the corresponding parameters as:
```
./deepsing.py temp/sample.mp3 temp/output --translator neural --path models/neural_translator_ --nid 20 --dynamic
```
If you do not define the *--nid* parameter, a random translator will be chosen. There are many more parameters to play with!

If you want to train *deepsing* from scratch, please download the corresponding datasets (please refer to [loaders](https://github.com/deepsing-ai/deepsing/tree/master/utils/loaders) ) and then ran the supplied train scripts.

If you use *deepsing* in your research, please cite our[ pre-print](https://arxiv.org/abs/1912.05654):

<pre>
@misc{deepsing,
    title={deepsing: Generating Sentiment-aware Visual Stories using Cross-modal Music Translation},
    author={Nikolaos Passalis and Stavros Doropoulos},
    year={2019},
    eprint={1912.05654},
    archivePrefix={arXiv},
}
</pre>


You are free to use the generated videos in any way you like, as long as you keep the deepsing logo.
