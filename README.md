# Posidonia-semantic-segmentation-multiclass

A Posidonia oceanica Semantic Segmentation model implemented in tensorflow.


<img src="vgg16fcn8/examples/159_00003.png" width="288"> <img src="vgg16fcn8/examples/159_00003_green.png" width="288">

<img src="vgg16fcn8/examples/v1_mix011.png" width="288"> <img src="vgg16fcn8/examples/v1_mix011_green.png" width="288">

cambiar por imagenes multiclass

Folder organization:

* preprocess: contains scripts to preprocess the images and ground thruts, resize, change extension, change color,...
* network: contains all the original network files, forked from https://github.com/MarvinTeichmann/KittiSeg.
* evaluation: contains scripts to binarize the output of the network, evaluate its performance and view the missclasified areas.
* uncertainty: contains scripts to calculate and evaluate the uncertainty areas of the network and the manual labelling process.

