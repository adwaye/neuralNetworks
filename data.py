import numpy as np
import gzip
import matplotlib.pyplot as plt
import cv2
import os
from build_image_data import _process_dataset

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255


"""
takes as input a gz-compressed ubyte object from http://yann.lecun.com/exdb/mnist/
and outputs it in folders that are named after the labels of each image
"""
def make_jpg_data(destinationLoc = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/train'
                  ,image_filename = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-images-idx3-ubyte.gz'
                  ,label_filename = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-labels-idx1-ubyte.gz'
                  ,num_images = 100,plot= False):
    bytestream_label = gzip.open(label_filename)
    bytestream_image = gzip.open(image_filename)
    bytestream_image.read(16)
    image_buffer = bytestream_image.read(IMAGE_SIZE * IMAGE_SIZE*num_images)
    bytestream_label.read(8)
    label_buffer = bytestream_label.read(IMAGE_SIZE * IMAGE_SIZE*num_images)
    if plot: plt.figure()
    for i in range(num_images):
        data_image = np.frombuffer(image_buffer[i*(IMAGE_SIZE * IMAGE_SIZE):(i+1)*IMAGE_SIZE * IMAGE_SIZE],
                                 dtype=np.uint8).astype(np.float32)

        lab        = label_buffer[i]
        saveLoc = os.path.join(destinationLoc,str(lab))
        if not os.path.isdir(saveLoc):
             os.makedirs(saveLoc)
        im         = data_image.reshape((IMAGE_SIZE,IMAGE_SIZE))
        cv2.imwrite(os.path.join(saveLoc,str(i) + '.jpeg') ,im)
        if plot:
            plt.imshow(im)
            plt.title(str(lab))
            plt.pause(0.01)
    if not os.path.isfile(os.path.join(destinationLoc,'label.txt')):
     with open(os.path.join(destinationLoc,'label.txt'), 'a') as fp:
         for name in range(10):
            fp.write(str(name)+'\n')
    print('done saving in '+destinationLoc)







if __name__=='__main__':
   print('extracting JPEG Mnist')
   train_image_filename = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/train-images-idx3-ubyte.gz'
   train_label_filename = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/train-labels-idx1-ubyte.gz'
   test_image_filename  = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-images-idx3-ubyte.gz'
   test_label_filename  = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-labels-idx1-ubyte.gz'
   make_jpg_data(destinationLoc = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/train',
                 image_filename=train_image_filename,label_filename=train_label_filename)
   make_jpg_data(destinationLoc = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/test',
                 image_filename=test_image_filename,label_filename=test_label_filename)
   trainLoc = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/train'
   print('')
   _process_dataset('./train',trainLoc,1,
                    os.path.join(trainLoc,'label.txt'))
   testLoc = '/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/test'
   _process_dataset('./validation',testLoc,1,
                    os.path.join(testLoc,'label.txt'))