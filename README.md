<div align='center'>
<h1>Deep-Learning-Intro-Assignments</h1>
</div>

**Description**

The purpose of this repository is to record the practical assignments from the Introduction to Deep Learning course at Tsinghua University, which teaches neural networks from scratch. This repository will be continuously updated, and currently includes the following assignments:

- HW1: Manual construction of gradient descent and Classification prediction for the MNIST dataset.(based on numpy, not pytorch)

- HWx2: Manually build MLP for classifying mnist data(based on numpy, not pytorch). The obtained loss decreases slowly and the accuracy is low, and the reason is still under investigation.(The error has been corrected because the fc layer calculated the gradient by dividing it by N)

- HW2: The main content of this assignment includes: familiarizing with basic operations of torch, building a neural network from scratch using torch, and attempting to solve regression problems using neural networks.

- HW3: The main content of HW3 is using CNNs for image classification. This assignment utilized two datasets, MNIST and CIFAR-10, and conducted several ablation studies on optimizers, Batch Normalization, and activation functions. Finally, ResNet was also implemented to assist with recognition. More details are covered in a report located in the HW3 folder.

- HWx3: The main content of Assignment x3 is to implement medical image segmentation using U-Net, with the dataset sourced from: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation. This task also includes more training details.

-HW4: The main task of HW4 is to use an RNN to classify sentence sentiment, employing the SST-1 dataset. In this task, a significant performance gap between the training set and the validation/test set emerged, the specific reasons for which are still under investigation.