# Splicing ViT Features for Semantic Appearance Transfer [[Project Page](https://github.io/tbd/)]
[![arXiv](https://img.shields.io/badge/arXiv-Splice-b31b1b.svg)](https://arxiv.org/abs/TBD)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.9.0-Red?logo=pytorch)
![teaser](imgs/teaser.png)


**Splice** is a method for semantic appearance transfer, as described in Splicing ViT Features for Semantic Appearance Transfer (link to paper).


>Given two input images—a source structure image and a target appearance image–our method generates a new image in which
the structure of the source image is preserved, while the visual appearance of the target image is transferred in a semantically aware manner.
That is, objects in the structure image are “painted” with the visual appearance of semantically related objects in the appearance image.
Our method leverages a self-supervised, pre-trained ViT model as an external semantic prior. This allows us to train our generator only on
a single input image pair, without any additional information (e.g., segmentation/correspondences), and without adversarial training. Thus,
our framework can work across a variety of objects and scenes, and can generate high quality results in high resolution (e.g., HD).


## Getting Started
### Installation

**Note:** The below installation will fail if run on something other than a CUDA GPU machine.
```
conda env create --file placeholder.yml
conda activate placeholder
```

System requirements
### System Requirements
- the following are a placeholder
- Python 3.7
- CUDA 10.2
- GPU w/ minimum 8 GB ram


### Run examples
Call the below shell scripts to generate example results.
```bash
# cow cow cow
./placeholder.sh
# ...
```
The outputs will be saved to `results/demo`, TBD.
#### Outputs
![plot](imgs/results.png)

## Placeholder
placeholder for something

## Citation
```
@article{text2mesh,
    author = {Tumanyan, Narek
              and Bar-Tal, Omer
              and Bagon, Shai
              and Dekel, Tali
              },
    title = {Splicing ViT Features for Semantic Appearance Transfer}, 
    journal = {arXiv preprint arXiv:TBD},
    year  = {2021}
}
```