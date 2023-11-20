# Pre-trained Model Robustness Against GAN-Based Poisoning Attack in Medical Imaging Analysis

## Description

This GitHub repository includes the code developed for my master's thesis. Additionally, 
it serves as the codebase corresponding to the following publication: 
[Link to the Paper](https://link.springer.com/chapter/10.1007/978-3-031-34111-3_26)

## Paper conclusion
In this paper, we conduct an experiment showing that the poisonous label attack can
compromise various CNN models used for medical image classification tasks. We use
conditional DCGAN with an average FID score of 33.58 to generate fake images of
normal and pneumonia lungs. After that, we update the five-trained CNN model with
generated images with the wrong label and observe the performance degradation. We
found that ConvNeXt is the most robust model among five others, followed by Resnet50v2. These might suggest that the technology in the Transformer and residual
connection, the main characteristic of ConvNeXt and ResNet, can be adapted to create a
more secure classification model in the medical images domain, and the simple architecture of stacking convolution layers model like VGG16 should be avoided. 


## Requirements

The code utilizes TensorFlow version 2.12 and the [clean-fid](https://github.com/GaParmar/clean-fid) library.

## Citation

Singkorapoom, P., Phoomvuthisarn, S. (2023). Pre-trained Model Robustness Against GAN-Based Poisoning Attack in Medical Imaging Analysis. In: Maglogiannis, I., Iliadis, L., MacIntyre, J., Dominguez, M. (eds) Artificial Intelligence Applications and Innovations. AIAI 2023. IFIP Advances in Information and Communication Technology, vol 675. Springer, Cham. https://doi.org/10.1007/978-3-031-34111-3_26







