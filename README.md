# Splicing ViT Features for Semantic Appearance Transfer [<a href="https://splice-vit.github.io" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-Splice-b31b1b.svg)](http://arxiv.org/abs/2201.00424)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.9.0-Red?logo=pytorch)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omerbt/Splice/blob/master/Splice.ipynb)
![teaser](imgs/teaser.png)


**Splice** is a method for semantic appearance transfer, as described in Splicing ViT Features for Semantic Appearance Transfer (<a href="http://arxiv.org/abs/2201.00424" target="_blank">link to paper</a>).


>Given two input images—a source structure image and a target appearance image–our method generates a new image in which
the structure of the source image is preserved, while the visual appearance of the target image is transferred in a semantically aware manner.
That is, objects in the structure image are “painted” with the visual appearance of semantically related objects in the appearance image.
Our method leverages a self-supervised, pre-trained ViT model as an external semantic prior. This allows us to train our generator only on
a single input image pair, without any additional information (e.g., segmentation/correspondences), and without adversarial training. Thus,
our framework can work across a variety of objects and scenes, and can generate high quality results in high resolution (e.g., HD).


## Getting Started
### Installation

```
git clone https://github.com/omerbt/Splice.git
pip install -r requirements.txt
```


### Run examples [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omerbt/Splice/blob/master/Splice.ipynb)


Run the following command to start training
```bash
python train.py --dataroot datasets/cows
```
Intermediate results will be saved to `<dataroot>/out/output.png` during optimization. The frequency of saving intermediate results is indicated in the `save_epoch_freq` flag of the configuration.

## Sample Results
![plot](imgs/results.png)

## Citation
```
@article{Splice2022,
    author = {Tumanyan, Narek
              and Bar-Tal, Omer
              and Bagon, Shai
              and Dekel, Tali
              },
    title = {Splicing ViT Features for Semantic Appearance Transfer}, 
    journal = {arXiv preprint arXiv:2201.00424},
    year  = {2022}
}
```