# Training a CNN with just 20 images


Author: 
Bhavan Vasu
Graduate Research Assistant, 
Real-time and computer vision lab,
Rochester Institute of Technology,
New York-14623
bxv7657@rit.edu

Requirements :
1) Numpy
2) matplotlib
3) tflearn
4) Opencv2

Add the training and testing samples into two folders in the current directory,named 'train' and 'test' respectively.

Alexnet was chosen for the implementation of a CatvsDog classifier using Tensorflow and python.(Check network graph for layer information)
The network is trained on 18 images and validated on 2 images during training.

The network is trained for just 12 epochs with a batch size of 16 with a learning rate of 1e-5.

The network manages to achieve a test accuracy of about 50-70%, measured from visual inspection of the 20 unknown test images. 
 

Run the 'catd.py' file for the CatvsDog classifier.

To test and train:

$ python catd.py

To run tensorboard for network visualization

$tensorboard --logdir="logs"


