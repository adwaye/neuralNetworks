import tensorflow as tf
from checkTfrecords import check_size,count_data, count_nClass
from utils import update_dir
import shutil
import os
from predict5 import *
from architectures import *
import numpy as np

tf.reset_default_graph()
batch_norm = False
weight_data = False
model = adnet_4
BATCH_SIZE = 400
N_EPOCHS = 200
CAPACITY = 2100
DEQUEUE = 100
DROP_RATE1 = 0.4
LEARNING_RATE = 0.00001
text_file = open("./data/label.txt", "r")
lines  = text_file.readlines()
nClass = len(lines)
trainFileName = './data/train-00000-of-00001'
testFileName = './data/validation-00000-of-00001'
shape = check_size()
DIMX_IM = shape[0]
DIMY_IM = shape[1]
kernel_size1 = [5,5]
kernel_size2 = [3,3]
kernel_size3 = [3,3]

def getImage(filename):
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

def trainModel(model,trainFileName="trainingDataCNN/train-00000-of-00001",testFileName="data/validation-00000-of-00001",
                   path_to_save='./trainedModel/',N_EPOCHS=100,LEARNING_RATE=0.0005,batch_norm=True,weight_data=False):
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
        label,image = getImage(trainFileName)
        # and similarly for the validation data
        vlabel,vimage = getImage(testFileName)

        imageBatch,labelBatch = tf.train.shuffle_batch(
            [image,label],batch_size=BATCH_SIZE,
            capacity=CAPACITY,
            min_after_dequeue=DEQUEUE,
            num_threads=2,
            allow_smaller_final_batch=True)

        # and similarly for the validation data
        vimageBatch,vlabelBatch = tf.train.shuffle_batch(
            [vimage,vlabel],batch_size=BATCH_SIZE,
            capacity=CAPACITY,
            min_after_dequeue=DEQUEUE,
            num_threads=2,
            allow_smaller_final_batch=True)

        with tf.name_scope('keep_probs'):
            keep_prob1 = tf.placeholder(tf.float32,name='FC1')


        trainFlag = tf.placeholder(tf.bool,name='trainFlag')

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32,[None,DIMY_IM* DIMX_IM],name='Patches')
    #        x_image = tf.reshape(x,[-1,DIMX_IM,DIMY_IM,1],name='Reshape_op')
            #tf.summary.image('image',x_image,max_outputs=3)
            y_ = tf.placeholder(tf.float32,[None,nClass],name='y-input')


        y = model(inputs=x,num_classes=nClass,is_training=trainFlag,drop_prob=keep_prob1,patchSize=[DIMX_IM,DIMY_IM],
                  batch_norm=batch_norm,kernel_size1=kernel_size1,kernel_size2=kernel_size2,kernel_size3=kernel_size3)

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

            #MARCUS READ THAT SHIT YOU WILL GET IT, YOU'RE SMART
            # start the threads used for reading files
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            #for i in range(nSteps):
            for i in range(nSteps):
                batch_xs,batch_ys = sess.run([imageBatch,labelBatch])
                if i % 100 == 0:  # Record summaries and test-set accuracy
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    vbatch_xs,vbatch_ys = sess.run([vimageBatch,vlabelBatch])
                    summary,acc = sess.run([merged,accuracy],feed_dict={x:vbatch_xs,y_:vbatch_ys,keep_prob1:1.0 - DROP_RATE1,
                                                                        trainFlag:False})
                    test_writer.add_summary(summary,i)
                    print('Accuracy at step %s: %s' % (i,acc))
                else:  # Record train set summaries, and train
                    if i % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary,_ = sess.run([merged,train_step],
                                             feed_dict={x:batch_xs,y_:batch_ys,keep_prob1:1.0 - DROP_RATE1,
                                                                        trainFlag:True},
                                             options=run_options,
                                             run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata,'step%03d' % i)
                        train_writer.add_summary(summary,i)
                        print('Adding run metadata for',i)
                        if not os.path.isdir(log_dir + '/savedModel'): os.mkdir(log_dir + '/savedModel')

                    else:  # Record a summary
                        summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y_:batch_ys,keep_prob1:1.0 -
                                                                                                              DROP_RATE1,
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
    log_dir = trainModel(trainFileName=trainFileName,testFileName=testFileName,model=model,
                         LEARNING_RATE=LEARNING_RATE,
                   N_EPOCHS=N_EPOCHS,batch_norm=batch_norm,weight_data=weight_data)


