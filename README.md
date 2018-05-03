# Posidonia-semantic-segmentation

A Posidonia oceanica Segmentation model implemented in tensorflow.

Folder organization:

* preprocess: contains scripts to preprocess the images and ground thruts, resize, change extension, change color,...
* network: contain all the original network files, forked from https://github.com/MarvinTeichmann/KittiSeg.
* evaluation: contains scripts to binarize the output of the network, evaluate its performance and view the missclasified areas.
* uncertainty: contains scripts to calculate and evaluate the uncertainty areas of the network and the manual labelling process.

# Citation

If you benefit from this code or dataset, please cite our paper:

```
@article{Miguel2018,
  title={Deep Semantic Segmentation for Posidonia Oceanica Meadows Identification},
  author={Miguel Martin-Abadal and Eric Guerrero-Font and Francisco Bonin-Font and Yolanda Gonzalez-Cid},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2018}
}
```
If you use the network code, cite the source paper from the forked Github
