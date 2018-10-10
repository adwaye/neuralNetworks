import os
import cv2
from checkTfrecords import *
from scipy.interpolate import griddata
from utils import _run
from sklearn.metrics import adjusted_rand_score


height = 32
width = 32

shape = check_size()
DIMX_IM = shape[0]
DIMY_IM = shape[1]
PRE_RESIZE = False
BATCH_SIZE = 300
N_EPOCHS = 1
CAPACITY = 250
DEQUEUE = 10
DROP_RATE = 0.4
N_DATA = count_data()
nSteps = BATCH_SIZE * N_EPOCHS
#text_file = open("./data/label.txt", "r")
#lines  = text_file.readlines()
nClass = 10#len(lines)








def predict_from_imLoc(dataLoc,modelLoc):
    #text_file = open("./data/label.txt","r")
    #lines = text_file.readlines()
    #nClass = len(lines)
    train_imgs = sorted([os.path.join(dataLoc,file) for file in os.listdir(dataLoc)],key=os.path.getctime,reverse=False)
    N = len(train_imgs)
    nBatch = int(N / BATCH_SIZE) + 1 * (N % BATCH_SIZE>0)
    def input_parser(img_path):
        # convert the label to one-hot encoding
        #one_hot = tf.one_hot(label, NUM_CLASSES)

        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file,channels=1)
        #img_decoded = tf.cast(tf.image.decode_image(img_file,channels=1),tf.float32)
        image = tf.image.convert_image_dtype(img_decoded,dtype=tf.float32)#.set_shape([33,33,1])
        image.set_shape([DIMX_IM,DIMY_IM,1])
        image = tf.reshape(1 - image,[DIMX_IM*DIMY_IM])

        return image

    tr_data = tf.data.Dataset.from_tensor_slices((train_imgs))
    tr_data = tr_data.map(input_parser).batch(batch_size=BATCH_SIZE).repeat()
    iterator= tr_data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        # initialize the iterator on the training data
        print('starting prediction')
        #saver = tf.train.import_meta_graph(modelName + '.meta')
        ckpt = tf.train.get_checkpoint_state(modelLoc)
        #saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess,ckpt.model_checkpoint_path)

        graph = tf.get_default_graph() #MARCUS I GET THE GRAPH
        ops = graph.get_operations() #MARCUS THESE ARE ALL THE OPS
        #Now, access the op that you want to run.
        #I NEED PLACEHOLDERS AS ALL THE OPS DEPEND ON THEM, THEY ARE IN A WAY THE INPUT TO MY GRAPH
        x = graph.get_tensor_by_name('input/Patches:0')
        y_ = graph.get_tensor_by_name('input/y-input:0')
        nClass = y_.shape[1].value
        try:
            prediction = graph.get_tensor_by_name('CL/final_pred/Softmax/Softmax:0')
        except KeyError:
            prediction = graph.get_tensor_by_name('CL/final_pred/Softmax/Softmax_2:0')
        keep_prob1 = graph.get_tensor_by_name('keep_probs/FC1:0')
         #we dont need true predictions for actual predictions
        # and similarly for the validation data
        for i in range(nBatch):
            try:
                print(i)
                xTest = sess.run(next_element)
                #sizeBatch = xTest.shape[0]
                if i == 0:
                    preds = sess.run(prediction,feed_dict={x:xTest,y_:np.random.random((xTest.shape[0],nClass)),
                                                           keep_prob1:1.0 - 0.4})
                else:
                    preds = np.concatenate( (preds,sess.run(prediction,feed_dict={x:xTest,y_:np.random.random((
                        xTest.shape[0],nClass)),
                                                                                  keep_prob1:1.0}) ) ,axis=0)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
    return preds, train_imgs

def parser(fullExample):
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
    #print()
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


    image = tf.reshape(1 - tf.image.rgb_to_grayscale(image),[DIMX_IM * DIMY_IM])
    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes.
    label = tf.stack(tf.one_hot(label - 1,nClass))
    #image.set_shape([height*width,1])
    #label = tf.cast(features['image/class/label'], tf.int32)
    return label,image,features['image/filename']

def predict_from_TfRecord(dataLoc,modelLoc):
    tr_data = tf.data.TFRecordDataset(dataLoc)
    tr_data = tr_data.map(parser).batch(batch_size=BATCH_SIZE).repeat()
    iterator= tr_data.make_one_shot_iterator()
    labelBatch,imageBatch,name = iterator.get_next()

    N = count_data(dataLoc)
    nBatch = int(N / BATCH_SIZE) + 1 * (N % BATCH_SIZE)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        # initialize the iterator on the training data

        print('starting prediction')
        ckpt = tf.train.get_checkpoint_state(modelLoc)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess,ckpt.model_checkpoint_path)

        #saver.restore(sess,tf.train.latest_checkpoint(modelLoc)) #MARCUS I RESTORE THE GMODEL
        graph = tf.get_default_graph() #MARCUS I GET THE GRAPH
        ops = graph.get_operations() #MARCUS THESE ARE ALL THE OPS
        #Now, access the op that you want to run.
        #I NEED PLACEHOLDERS AS ALL THE OPS DEPEND ON THEM, THEY ARE IN A WAY THE INPUT TO MY GRAPH
        x = graph.get_tensor_by_name('input/Patches:0') #MARCUS THIS IS MY INPUT PLACEHOLDER
        y_ = graph.get_tensor_by_name('input/y-input:0') #MARCUS SAME SHIT HERE PLACEHOLDE
        nClass = y_.shape[1].value
        try:
            prediction = graph.get_tensor_by_name('CL/final_pred/Softmax/Softmax:0')
        except KeyError:
            prediction = graph.get_tensor_by_name('CL/final_pred/Softmax/Softmax_2:0')
        keep_prob1 = graph.get_tensor_by_name('keep_probs/FC1:0')
         #we dont need true predictions for actual predictions
        # and similarly for the validation data
        for i in range(nBatch):
            try:

                print(i)
                xTest = sess.run(imageBatch)
                if i == 0:
                    coords = sess.run(name)
                    preds  = sess.run(prediction,feed_dict={x:xTest,y_:np.random.random((xTest.shape[0],
                                                                                                      nClass)),
                                                           keep_prob1:1.0})
                else:
                    preds = np.concatenate( (preds,sess.run(prediction,feed_dict={x:xTest,y_:np.random.random((xTest.shape[0],2)),
                                                                                  keep_prob1:1.0}) ) ,axis=0)
                    coords= np.concatenate((coords,sess.run(name)))
                #input('nextelem')
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        coord.request_stop()
        #coord.join(threads)
    splitter = lambda t: t.decode('utf-8')#.split('.')
    vfunc = np.vectorize(splitter)
    coordsFlat = vfunc(coords)

    return preds, coordsFlat


if __name__ == '__main__':
        #"""
        tf.reset_default_graph()
        if os.path.isdir('./trainedModel'):
            print('found the following models')
            print(sorted(os.listdir('trainedModel')))
            modelLoc = './trainedModel/1/savedModel'
            modelName = modelLoc + '/model-5001'
            print('predicting shit')
            #preds = predict_from_imLoc('./validationData/2/True/')
            #preds, names = predict_from_TfRecord('./data/eval-00000-of-00001',modelLoc=modelLoc)
            #coords = np.loadtxt('./validationData/1/coords.txt')
            im = cv2.imread('./validationData/testIm4.png',0)
            #im = 1-cv2.imread('./validationData/CPSA0054h2012.png',0)
            patchSize = 33
        else:
            print('no trained model found in ./trainedModel')
        #preds,train_imgs = predict_from_imLoc(dataLoc=,modelLoc=modelLoc)
        #todo: need to have some test images to predict on and also to have some sort of demo of the predictions


