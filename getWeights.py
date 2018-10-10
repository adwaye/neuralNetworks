
import tensorflow as tf


slim = tf.contrib.slim
import tensorflow.contrib.slim.nets as nets
import sys
sys.path.append('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/discriminativeModel/')
import matplotlib.pyplot as plt
plt.ion()
import scipy.io as sio
tf.reset_default_graph()

convList = ['vgg_16/conv1/conv1_1','vgg_16/conv1/conv1_2','vgg_16/conv2/conv2_1','vgg_16/conv2/conv2_2',
            'vgg_16/conv3/conv3_1','vgg_16/conv3/conv3_2','vgg_16/conv3/conv3_3','vgg_16/conv4/conv4_1',
            'vgg_16/conv4/conv4_2','vgg_16/conv4/conv4_3','vgg_16/conv5/conv5_1','vgg_16/conv5/conv5_2',
            'vgg_16/conv5/conv5_3']

ckpt = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/artStuff/models/vgg_16/vgg_16.ckpt'
varList = []
w1_1 = tf.get_variable("vgg_16/conv1/conv1_1/weights", [3,3,3,64], initializer = tf.zeros_initializer)
varList = varList + [w1_1]
#"""
b1_1 = tf.get_variable("vgg_16/conv1/conv1_1/biases", [64], initializer = tf.zeros_initializer)
varList = varList + [b1_1]
w1_2 = tf.get_variable("vgg_16/conv1/conv1_2/weights", [3,3,64,64], initializer = tf.zeros_initializer)
varList = varList + [w1_2]
b1_2 = tf.get_variable("vgg_16/conv1/conv1_2/biases", [64], initializer = tf.zeros_initializer)
varList = varList + [b1_2]

w2_1 = tf.get_variable("vgg_16/conv2/conv2_1/weights", [3,3,64,128], initializer = tf.zeros_initializer)
varList = varList + [w2_1]
b2_1 = tf.get_variable("vgg_16/conv2/conv2_1/biases", [128], initializer = tf.zeros_initializer)
varList = varList + [b2_1]
w2_2 = tf.get_variable("vgg_16/conv2/conv2_2/weights", [3,3,128,128], initializer = tf.zeros_initializer)
varList = varList + [w2_2]
b2_2 = tf.get_variable("vgg_16/conv2/conv2_2/biases", [128], initializer = tf.zeros_initializer)
varList = varList + [b2_2]

w3_1 = tf.get_variable("vgg_16/conv3/conv3_1/weights", [3,3,128,256], initializer = tf.zeros_initializer)
varList = varList + [w3_1]
b3_1 = tf.get_variable("vgg_16/conv3/conv3_1/biases", [256], initializer = tf.zeros_initializer)
varList = varList + [b3_1]
w3_2 = tf.get_variable("vgg_16/conv3/conv3_2/weights", [3,3,256,256], initializer = tf.zeros_initializer)
varList = varList + [w3_2]
b3_2 = tf.get_variable("vgg_16/conv3/conv3_2/biases", [256], initializer = tf.zeros_initializer)
varList = varList + [b3_2]
w3_3 = tf.get_variable("vgg_16/conv3/conv3_3/weights", [3,3,256,256], initializer = tf.zeros_initializer)
varList = varList + [w3_3]
b3_3 = tf.get_variable("vgg_16/conv3/conv3_3/biases", [256], initializer = tf.zeros_initializer)
varList = varList + [b3_3]


w4_1 = tf.get_variable("vgg_16/conv4/conv4_1/weights", [3,3,256,512], initializer = tf.zeros_initializer)
varList = varList + [w4_1]
b4_1 = tf.get_variable("vgg_16/conv4/conv4_1/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b4_1]
w4_2 = tf.get_variable("vgg_16/conv4/conv4_2/weights", [3,3,512,512], initializer = tf.zeros_initializer)
varList = varList + [w4_2]
b4_2 = tf.get_variable("vgg_16/conv4/conv4_2/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b4_2]
w4_3 = tf.get_variable("vgg_16/conv4/conv4_3/weights", [3,3,512,512], initializer = tf.zeros_initializer)
varList = varList + [w4_3]
b4_3 = tf.get_variable("vgg_16/conv4/conv4_3/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b4_3]

w5_1 = tf.get_variable("vgg_16/conv5/conv5_1/weights", [3,3,512,512], initializer = tf.zeros_initializer)
varList = varList + [w5_1]
b5_1 = tf.get_variable("vgg_16/conv5/conv5_1/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b5_1]
w5_2 = tf.get_variable("vgg_16/conv5/conv5_2/weights", [3,3,512,512], initializer = tf.zeros_initializer)
varList = varList + [w5_2]
b5_2 = tf.get_variable("vgg_16/conv5/conv5_2/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b5_2]
w5_3 = tf.get_variable("vgg_16/conv5/conv5_3/weights", [3,3,512,512], initializer = tf.zeros_initializer)
varList = varList + [w5_3]
b5_3 = tf.get_variable("vgg_16/conv5/conv5_3/biases", [512], initializer = tf.zeros_initializer)
varList = varList + [b5_3]
#"""

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver()

# Use the saver object normally after that.
listOfKern     = []
listOfKernName = []
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  saver.restore(sess, ckpt)
  for i in range(len(varList)):
    listOfKern     = listOfKern+ [sess.run(varList[i])]
    listOfKernName = listOfKernName + [varList[i].name]

  sio.savemat( 'vgg_16_conv.mat',{'names':listOfKernName,'weights':listOfKern,'convList':convList})













