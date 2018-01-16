"""Code to detect pips in a dice using c++ and opencv 
Author:Bhavan Vasu,
Graduate research assistant,
Real time and computer vision lab,
Rochester Institute of technology,New york-14623
email:bxv7657@rit.edu"""
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
import os     
import random 
import cv2
from tflearn.data_utils import image_preloader
import math

img_dir = '/home/bxv7657/NASCENT/cat/train' #path to folder containing all your training images
train_tx = '/home/bxv7657/NASCENT/cat/training_data.txt'
val_tx = '/home/bxv7657/NASCENT/cat/validation_data.txt'
train_ratio=0.9
val_ratio=0.1

#extracting image names
filepat = os.listdir(img_dir)
random.shuffle(filepat)

#calculating total number of training images
train_len=len(filepat)

#Generating training labels
fr = open(train_tx, 'w')
train_files=filepat[0: int(train_ratio*train_len)]
for filename in train_files:
    if filename[0:3] == 'cat':
        fr.write(img_dir + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        fr.write(img_dir + '/'+ filename + ' 1\n')

fr.close()

#Generating validation labels
fr = open(val_tx, 'w')
valid_files=filepat[int(math.ceil((train_ratio)*train_len)):train_len]
for filename in valid_files:
    if filename[0:3] == 'cat':
        fr.write(img_dir + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        fr.write(img_dir + '/'+ filename + ' 1\n')
fr.close()

X_train, Y_train = image_preloader(train_tx, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
X_val, Y_val = image_preloader(val_tx, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
print ("Number of training images {}".format(len(X_train)))
print ("Number of validation images {}".format(len(X_val)))
print ("number of classes: {}",len(Y_train[1]))

#input image
with tf.name_scope("input"):
 x=tf.placeholder(tf.float32,shape=[None,56,56,3] , name='input_image') 
#input label
 y_=tf.placeholder(tf.float32,shape=[None, 2] , name='input_class')

input_im=x
#convolutional layer 1
with tf.name_scope("Layer_1"):
 conv_1=tflearn.layers.conv.conv_2d(input_im, nb_filter=64, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu', regularizer="L2", )
with tf.name_scope("Maxpool_1"):
#max pooling layer 1
 out_1=tflearn.layers.conv.max_pool_2d(conv_1, 2)

#convolutional layer 2
with tf.name_scope("Layer_2"):
 conv_2=tflearn.layers.conv.conv_2d(out_1, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2")
with tf.name_scope("Maxpool_2"):
 out_2=tflearn.layers.conv.max_pool_2d(conv_2, 2)
#convolutional layer 3
with tf.name_scope("Layer_3"):
 conv_3=tflearn.layers.conv.conv_2d(out_2, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2")
with tf.name_scope("Maxpool_3"):
 out_3=tflearn.layers.conv.max_pool_2d(conv_3, 2)
#fully connected layer1
with tf.name_scope("FC1"):
 fc1= tflearn.layers.core.fully_connected(out_3, 4096, activation='relu' )
 fc1_dropout = tflearn.layers.core.dropout(fc1, 1)
#fully connected layer2
with tf.name_scope("FC2"):
 fc2= tflearn.layers.core.fully_connected(fc1_dropout, 4096, activation='relu' )
 fc2_dropout = tflearn.layers.core.dropout(fc2, 1)
#softmax layer output
with tf.name_scope("Softmax"):
 y_predicted = tflearn.layers.core.fully_connected(fc2_dropout, 2, activation='softmax')

#loss function
with tf.name_scope("Loss"):
 cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))
#optimiser 
with tf.name_scope("train"):
 train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
#tensorboard file writer location
writer=tf.summary.FileWriter("logs/",sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
g = tf.get_default_graph()
[op.name for op in g.get_operations()]

epoch=12
batch_size=16
itr_epo=len(X_train)//batch_size

itr_epo
n_val=len(X_val)
print epoch
print np.shape(X_train)
# Training the network for n epocs
for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))
    
    previous_batch=0
    for i in range(itr_epo):
        current_batch=previous_batch+batch_size
        x_input=X_train[previous_batch:current_batch] 
        x_images=np.reshape(x_input,[batch_size,56,56,4])
        x_images=x_images[:,:,:,0:3]
        
        y_input=Y_train[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,2])
        previous_batch=previous_batch+batch_size
        
        _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y_: y_label})
        if i % 100==0 :
            print ("Training loss : {}" .format(loss))
#Montioring loss for convergence
            
    print np.shape(X_val[0:n_val])
    x_val_images=np.reshape(X_val[0:n_val],[n_val,56,56,4])
    x_val_images= x_val_images[:,:,:,0:3];
    y_val_labels=np.resize(Y_val[0:n_val],[n_val,2])
    Accuracy_val=sess.run(accuracy,
                           feed_dict={
                        x: x_val_images ,
                        y_: y_val_labels
                      })    
    Accuracy_val=round(Accuracy_val*100,2)
#Monitoring validation accuracy for overfitting
    print("Accuracy :Validation_accuracy {} % " .format(Accuracy_val))
saver = tf.train.Saver()
save_path=saver.save(sess,"/home/bxv7657/NASCENT/cat/model.ckpt");

def process_img1(img):
        img=img/np.max(img).astype(float) 
        img=np.reshape(img, [1,56,56,3])      
        return img
test_image = []
   
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
#loading test images
image_list=load_images_from_folder('/home/bxv7657/NASCENT/cat/test');
print np.shape(image_list[1] )
for i in range(len(os.listdir('/home/bxv7657/NASCENT/cat/test'))):
    test_image=image_list[i];
    test_image=np.resize(image_list[i],[56,56,3])
    test_i= process_img1(test_image)
    #predicting class for test image
    predicted_array= sess.run(y_predicted, feed_dict={x: test_i})
    predicted_class= np.argmax(predicted_array)
    if predicted_class==0:
     print os.listdir('/home/bxv7657/NASCENT/cat/test')[i]+"-cat"
    if predicted_class==1:
     print os.listdir('/home/bxv7657/NASCENT/cat/test')[i]+"-dog"
