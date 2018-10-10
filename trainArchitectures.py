import tensorflow as tf
from checkTfrecords import check_size,count_data, count_nClass
from utils import update_dir, del_all_flags
import shutil
import os
from predict5 import *
from architectures import *
import numpy as np

tf.reset_default_graph()
del_all_flags(tf.flags.FLAGS)



tf.app.flags.DEFINE_integer('kernel_size_1',5,
                           'Size of first convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_2',5,
                           'Size of second convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_3',3,
                           'Size of third convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_4',3,
                           'Size of fourth convolutional kernel')
#flags: can be used to change parameters by passing --name_of_flag value in the command line while launching the script
#the name of the flag is given in the first argument of the define_flag function
#batching parameters
tf.app.flags.DEFINE_integer('batch_size',400,
                           'size of input batches')
tf.app.flags.DEFINE_integer('capacity',2100,
                           'length of data queue for the input pipeline')
tf.app.flags.DEFINE_integer('dequeue',100,
                           'size that queue reaches before adding more data to it')

#training parameters
tf.app.flags.DEFINE_float('drop_rate',0.4,
                           'ratio of neurons that are dropped in the drop out layer')
tf.app.flags.DEFINE_float('l_rate',0.0001,
                           'learning rate for the gradient descent optimiser')
tf.app.flags.DEFINE_integer('n_epochs',200,
                           'learning rate for the gradient descent optimiser')

#data location
tf.app.flags.DEFINE_string('training_data','./data/train-00000-of-00001',
                           'path to the TFRecord file containing the training data')
tf.app.flags.DEFINE_string('test_data','./data/validation-00000-of-00001',
                           'path to the TFRecord file containing the validation data')


#booleans
tf.app.flags.DEFINE_bool('batch_norm',False,
                           'boolean: if true, batch normalisation is applied between layers')
tf.app.flags.DEFINE_bool('weight_data',False,
                           'boolean: if true, the loss function is weighted according to the data split so that the '
                           'class with less data is given more weight')


#model choice
tf.app.flags.DEFINE_integer('model',4,
                           'integer for choosing the architecture, there are 8 in the architectures.py file')




FLAGS = tf.app.flags.FLAGS

batch_norm = FLAGS.batch_norm
weight_data = FLAGS.weight_data
model = model_library[str(FLAGS.model)]
BATCH_SIZE = FLAGS.batch_size
N_EPOCHS = FLAGS.n_epochs
CAPACITY = FLAGS.capacity
DEQUEUE = FLAGS.dequeue
DROP_RATE = FLAGS.drop_rate
LEARNING_RATE = FLAGS.l_rate
#text_file = open("./data/label.txt", "r")
#lines  = text_file.readlines()
trainTFRecord = FLAGS.training_data
testTFRecord  = FLAGS.test_data
trainLoc      = './MNIST-data/images/train'
testLoc       = './MNIST-data/images/test'
nClass = len(count_nClass(testTFRecord))
shape = check_size(testTFRecord)
DIMX_IM    = shape[0]
DIMY_IM    = shape[1]
IMAGE_SIZE = DIMY_IM





"""
parser that reads the tfrecord file and returns one example of a label, image pair, this parser is used to create a
dataset object in to be fed in a CNN as an input
input
tfrecords_filename: path to tfrecord file containing data as generated by build_image_data._process_dataset
output
example : tuple with the first element being the class that the image belongs to and where the 2nd element is the image
"""

def getTFRecordImage(filename):
    shape = check_size()
    DIMX_IM = shape[0]
    DIMY_IM = shape[1]
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename],num_epochs=None)

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key,fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height':tf.FixedLenFeature([],tf.int64),
            'image/width':tf.FixedLenFeature([],tf.int64),
            'image/colorspace':tf.FixedLenFeature([],dtype=tf.string,default_value=''),
            'image/channels':tf.FixedLenFeature([],tf.int64),
            'image/class/label':tf.FixedLenFeature([],tf.int64),
            'image/class/text':tf.FixedLenFeature([],dtype=tf.string,default_value=''),
            'image/format':tf.FixedLenFeature([],dtype=tf.string,default_value=''),
            'image/filename':tf.FixedLenFeature([],dtype=tf.string,default_value=''),
            'image/encoded':tf.FixedLenFeature([],dtype=tf.string,default_value='')
            #'image/encoded': tf.VarLenFeature([], dtype=tf.string)
        })

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg',[image_buffer],None):
        # decode
        image = tf.image.decode_jpeg(image_buffer,channels=1)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)#.set_shape([33,33,1])
        image.set_shape([DIMX_IM,DIMY_IM,1])

    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel.
    # the "1-.." part inverts the image, so that the background is black.
    #image = tf.reshape(1 - tf.image.rgb_to_grayscale(image),[DIMX_IM * DIMY_IM])

    image = tf.reshape(1 -image,[DIMX_IM * DIMY_IM])
    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes.
    label = tf.stack(tf.one_hot(label - 1,nClass))
    return label,image





"""
creates batches to be fed into a CNN from a TFRecordsFile
input: location of TFRecord file, needs to have the format generated by build_image_data.py
       e.g: data_loc/0 has images belonging to class 0 etc etc
output: list of tensors with first element being a batch of labels and the second element being a batch of images
"""
def input_from_TFRecords(fileName):
    label,image = getTFRecordImage(fileName)
    # and similarly for the validation data


    imageBatch,labelBatch = tf.train.shuffle_batch(
        [image,label],batch_size=BATCH_SIZE,
        capacity=CAPACITY,
        min_after_dequeue=DEQUEUE,
        num_threads=1,
        allow_smaller_final_batch=True)

    return labelBatch, imageBatch


"""
input parser that reads one image and returns the label and image tensor
input: string pointing to the location of one image example,
note that the folder containing the image need to have the label as a name
output: an image example to be handled by dataset class in input pipeline
"""
def getJPGImage(img_path):
    # read the img from file
    label = int(os.path.split(img_path)[1][-1])
    label = tf.stack(tf.one_hot(label,nClass))
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file,channels=1)
    #img_decoded = tf.cast(tf.image.decode_image(img_file,channels=1),tf.float32)
    image = tf.image.convert_image_dtype(img_decoded,dtype=tf.float32)#.set_shape([33,33,1])
    image.set_shape([DIMX_IM,DIMY_IM,1])
    image = tf.reshape(1 - image,[DIMX_IM*DIMY_IM])

    return label,image




"""
creates batches to be fed into a CNN from a location containing folders with jpeg images in
input: location of data. needs to contain folders whose names are labels correspond to the class the images
       in the folders belong to.
       e.g: data_loc/0 has images belonging to class 0 etc etc
output: list of tensors with first element being a batch of labels and the second element being a batch of images
"""
def input_from_im_loc(data_loc='/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/images/train'):
    dirs = os.listdir(data_loc)
    train_imgs = []
    for lab in dirs:
        dir = os.path.join(data_loc,lab)
        if os.path.isdir(dir):
            train_imgs = train_imgs+ [os.path.join(dir,file) for file in os.listdir(dir)]

    tr_data = tf.data.Dataset.from_tensor_slices((train_imgs))
    tr_data = tr_data.map(getJPGImage()).batch(batch_size=BATCH_SIZE).repeat()
    iterator = tr_data.make_one_shot_iterator()
    labelBatch, imageBatch = iterator.get_next()
    return labelBatch, imageBatch



#img_path is a Q
#todo: figure out how to create batches out of ubytes directly
#in a tfrecord file, the process of decoding the bytes to a jpeg is the same as the one in the ubyte image,
#however, the tfrecord file already has an iterative structure native to tensorflow. That is, it is easy from a
# syntax point of view to iterate across image examples. in the ubyte the process that needs to be made iterable is
# the one where the ubyte array is sliced into pieces having the size of the flattened
#array
def getUBYTEImage(img_path='/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-images-idx3-ubyte.gz'
                 ,lab_path='/home/amr62/Documents/github examples/neuralNetworks/MNIST-data/t10k-labels-idx1-ubyte.gz'
                  ):
    pass
    #reader = tf.FixedLengthRecordReader(record_bytes=IMAGE_SIZE*IMAGE_SIZE)
    #key,value = reader.read(img_path)
    #record_bytes = tf.decode_raw(value,tf.uint8)


    # Convert from [depth, height, width] to [height, width, depth].
    #result = tf.reshape(record_bytes,[IMAGE_SIZE,IMAGE_SIZE])
    #return result



def trainModel(model,trainFileName,testFileName,
               path_to_save='./trainedModel/',N_EPOCHS=100,LEARNING_RATE=0.0005,batch_norm=True,weight_data=False,
               use_TFRecord=True
               ):
    shape = check_size(trainFileName)
    DIMX_IM = shape[0]
    DIMY_IM = shape[1]
    N_DATA = count_data(trainFileName)
    nSteps = int((N_DATA / BATCH_SIZE) * N_EPOCHS)
    #nSteps = 1000
    #nClass = 2
    #train data
    g1 = tf.Graph()
    with g1.as_default():
        if use_TFRecord:
            labelBatch,imageBatch = input_from_TFRecords(trainFileName)
            vlabelBatch,vimageBatch = input_from_TFRecords(testFileName)
        else:
            #this is if you want to load images directly from a location with
            #labelled folders containing jpegs
            labelBatch,imageBatch = input_from_im_loc(trainLoc)
            vlabelBatch,vimageBatch = input_from_im_loc(testLoc)


        with tf.name_scope('keep_probs'):
            keep_prob1 = tf.placeholder(tf.float32,name='FC1')


        trainFlag = tf.placeholder(tf.bool,name='trainFlag')

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32,[None,DIMY_IM* DIMX_IM],name='Patches')
    #        x_image = tf.reshape(x,[-1,DIMX_IM,DIMY_IM,1],name='Reshape_op')
            #tf.summary.image('image',x_image,max_outputs=3)
            y_ = tf.placeholder(tf.float32,[None,nClass],name='y-input')


        y = model(inputs=x,num_classes=nClass,is_training=trainFlag,drop_prob=keep_prob1,imSize=[DIMX_IM,DIMY_IM],
                  batch_norm=batch_norm)

        with tf.name_scope('cross_entropy'):
            if weight_data:
                weights = count_nClass(trainFileName)
                denom   = np.sum(weights)
                weights = 1/(weights/denom)
                weights = tf.constant(weights,dtype=tf.float32)
                cross_entropy = tf.losses.softmax_cross_entropy(weights*y_,y)
            else:
                cross_entropy = tf.losses.softmax_cross_entropy(y_,y)
            tf.summary.scalar('cross_entropy',cross_entropy)

        # define training step which minimises cross entropy
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy,
                                                                                      global_step=tf.train.get_global_step())

        # argmax gives index of highest entry in vector (1st axis of 1D tensor)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1),tf.argmax(y_,1))
            with tf.name_scope('accuracy'):
                # get mean of all entries in correct prediction, the higher the better
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)


        log_dir = update_dir(path_to_save)
        log_dir_board = os.path.join(log_dir , 'tensorboard')
        log_dir_save  = os.path.join(log_dir , 'savedModel/model')
        merged = tf.summary.merge_all()

        # run the session
        # start training and record shit
        # initialize the variables
        saver = tf.train.Saver(max_to_keep=8)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter(log_dir_board + '/train',sess.graph)
            test_writer = tf.summary.FileWriter(log_dir_board + '/test')
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            # start the threads used for reading files
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            #for i in range(nSteps):
            nSteps = 0
            for i in range(1):
                batch_xs,batch_ys = sess.run([imageBatch,labelBatch])
                if i % 100 == 0:  # Record summaries and test-set accuracy
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    vbatch_xs,vbatch_ys = sess.run([vimageBatch,vlabelBatch])
                    summary,acc = sess.run([merged,accuracy],feed_dict={x:vbatch_xs,y_:vbatch_ys,keep_prob1:1.0 - DROP_RATE,
                                                                        trainFlag:False})
                    test_writer.add_summary(summary,i)
                    print('Accuracy at step %s: %s' % (i,acc))
                else:  # Record train set summaries, and train
                    if i % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary,_ = sess.run([merged,train_step],
                                             feed_dict={x:batch_xs,y_:batch_ys,keep_prob1:1.0 - DROP_RATE,
                                                                        trainFlag:True},
                                             options=run_options,
                                             run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata,'step%03d' % i)
                        train_writer.add_summary(summary,i)
                        print('Adding run metadata for',i)
                        if not os.path.isdir(log_dir + '/savedModel'): os.mkdir(log_dir + '/savedModel')

                    else:  # Record a summary
                        summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y_:batch_ys,keep_prob1:1.0 -
                                                                                                              DROP_RATE,
                                                                        trainFlag:True})
                        train_writer.add_summary(summary,i)

                if i % 1000 == 0:
                    save_path = saver.save(sess,log_dir_save,global_step=i + 1)

            save_path = saver.save(sess,log_dir_save,global_step=i + 1)
            train_writer.close()
            test_writer.close()
            # finalise
            coord.request_stop()
            coord.join(threads)
    return os.path.join(log_dir,'savedModel')


if __name__=='__main__':
    print('Architecture\n'+architecture_library[str(FLAGS.model)])
    log_dir = trainModel(trainFileName=trainTFRecord,testFileName=testTFRecord,model=model,
                         LEARNING_RATE=LEARNING_RATE,
                         N_EPOCHS=N_EPOCHS,batch_norm=batch_norm,weight_data=weight_data)


