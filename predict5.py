from _data import *
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
text_file = open("./data/label.txt", "r")
lines  = text_file.readlines()
#nClass = len(lines)




def predict_from_im(im,modelLoc,skip=1,patchSize=23,rowSize=90,colSize=100):
    im = np.reshape(im,(1,im.shape[0],im.shape[1],1))
    im = 1-im/np.max(im)
    nX = int(im.shape[1]/rowSize)
    if im.shape[1]%rowSize!=0:
        nX = nX+1
    nY = int(im.shape[2] / colSize)
    if im.shape[2]%colSize!=0:
        nY = nY+1
    #nX = 5
    #nY = 5
    ksizes = [1,patchSize,patchSize,1]
    bitSize = 1000
    strides = [1,skip,skip,1]
    rates = [1,1,1,1]
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
            prediction = graph.get_tensor_by_name('logits/logits/BiasAdd:0')
        keep_prob1 = graph.get_tensor_by_name('keep_probs/FC1:0')
        trainFlag = graph.get_tensor_by_name('trainFlag:0')
         #we dont need true predictions for actual predictions
        # and similarly for the validation data
        for i in range(nX):
            print('row '+str(i))
            for j in range(nY):
                print(j)
                offsetStartx = int(patchSize/2)
                offsetStarty = int(patchSize/2)
                if i==0: offsetStartx = 0
                if j==0: offsetStarty = 0
                offsetEndx = int(patchSize/2)+1
                offsetEndy = int(patchSize/2)+1
                if i==nX-1: offsetEndx=0
                if j==nY-1: offsetEndy=0
                startX = i*rowSize-offsetStartx
                startY = j*colSize-offsetStarty
                endX   = (i+1)*rowSize+offsetEndx #TODO: CHECK IF ADDING A 1 AT THE END FIXES IT, PYTHON INDEXING,
                # did that, need to see if it works
                # NEEDS TO END [0,1,2,3] IS 0:4
                endY   = (j+1)*colSize+offsetEndy
                if i==nX-1:
                    endX = im.shape[1]
                if j==nY-1:
                    endY = im.shape[2]

                cropIm   = im[:,startX:endX,startY:endY,:]
                xDim = cropIm.shape[1]
                yDim = cropIm.shape[2]
                tf_patch = tf.extract_image_patches(cropIm,ksizes=ksizes,strides=strides,rates=rates,padding='SAME')
                xTest    = sess.run(tf.reshape(tf_patch,[xDim*yDim,patchSize**2]))
                nBits = xDim*yDim/bitSize
                nTraverse = int(nBits)
                if xTest.shape[0]%bitSize !=0:
                    nTraverse = nTraverse + 1
                for k in range(nTraverse):
                    if k<nBits:
                        input = xTest[k*bitSize:bitSize*(k+1),:]
                    else:
                        input = xTest[nBits*bitSize:,:]
                    if k==0:
                        preds = sess.run(tf.nn.softmax(prediction),feed_dict={x:input,y_:np.random.random((
                            xTest.shape[0],
                                                                                                 nClass)),
                                                               keep_prob1:1.0 - 0.4,trainFlag:False})
                    else:
                        res = sess.run(tf.nn.softmax(prediction),feed_dict={x:input,y_:np.random.random((xTest.shape[0],
                                                                                                 nClass)),
                                                             keep_prob1:1.0 - 0.4,trainFlag:False})
                        preds = np.concatenate((preds,res),axis=0)
                if j==0:
                    predsResh = preds.reshape((xDim,yDim,
                                           nClass))[offsetStartx:rowSize+offsetStartx,
                                offsetStarty:colSize+offsetStarty,:]
                else:
                    predsResh = np.concatenate((predsResh,preds.reshape((xDim,yDim,nClass))[offsetStartx:rowSize+offsetStartx,offsetStarty:colSize+offsetStarty,:]  ),
                                               axis=1)
            if i ==0:
                finalPred = predsResh
            else:
                finalPred = np.concatenate((finalPred,predsResh),axis=0)


    return finalPred #TODO: REBUILD THE IMAGE USING THE CENTRE PIXEL OF THE PATCHES AS OUTPUTS, AND CHECK IF I GET
    # THE ORIGINAL IMAGE



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
    return preds

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
    splitter = lambda t: t.decode('utf-8')[:-5]#.split('.')
    vfunc = np.vectorize(splitter)
    coordsFlat = vfunc(coords)

    return preds, coordsFlat

def turn_names_to_coords(names):
    for i in range(len(names)):
        nameList = names[i].split('_')
        if i==0:
            coords = np.array( [[int(nameList[2]),int(nameList[3])]] )
        else:
            coords = np.concatenate( (coords,np.array( [[int(nameList[2]),int(nameList[3])]] ) ),axis = 0 )
    return coords

if __name__ == '__main__':
        #"""
        tf.reset_default_graph()
        modelLoc = './trainedModel/33/savedModel'
        modelName = modelLoc + '/model-5001'
        print('predicting shit')
        #preds = predict_from_imLoc('./validationData/2/True/')
        #preds, names = predict_from_TfRecord('./data/eval-00000-of-00001',modelLoc=modelLoc)
        #coords = np.loadtxt('./validationData/1/coords.txt')
        im = cv2.imread('./validationData/testIm4.png',0)
        #im = 1-cv2.imread('./validationData/CPSA0054h2012.png',0)
        patchSize = 33
        preds = predict_from_im(im,modelLoc,patchSize=patchSize)
        predMax = np.argmax(preds,axis=2)
        #coords = turn_names_to_coords(names)
        #plt.figure()
        #grid_z0 = turn_into_potential(preds[:,0],coords,im,method='cubic')
        ncols = preds.shape[2]+1
        fig,ax = plt.subplots(nrows=1,ncols=ncols)
        ax[0].imshow(im)
        ax[0].set_title('image')
        for i in range(1,ncols):
            ax[i].imshow(preds[:,:,i-1])
            ax[i].set_title('class '+ str(i)+ ' probs')
        fig.savefig(os.path.join(modelLoc[:-10],'testIm4.png'))
        #im = cv2.imread('./validationData/testIm2.png',0)
        #im = 1-cv2.imread('./validationData/CPSA0054h2012.png',0)
        """
        preds = predict_from_im(im,modelLoc,patchSize=patchSize)
        predMax = np.argmax(preds,axis=2)
        #coords = turn_names_to_coords(names)
        #plt.figure()
        #grid_z0 = turn_into_potential(preds[:,0],coords,im,method='cubic')
        ncols = preds.shape[2]+1
        fig,ax = plt.subplots(nrows=1,ncols=ncols)
        ax[0].imshow(im)
        ax[0].set_title('image')
        for i in range(1,ncols):
            ax[i].imshow(preds[:,:,i-1])
            ax[i].set_title('class '+ str(i)+ ' probs')
        fig.savefig(os.path.join(modelLoc[:-10],'testIm2.png'))
        #plt.colorbar(predMax,ax=ax[2])


        imName   = 'CPSA0004h2012'
        im       =  cv2.imread(os.path.join('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/ARxrays/histNorm/1/',
                                            imName+'.png'),0)
        curveLoc = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/mergedCurvePoints/'
        Nx = im.shape[0]
        Ny = im.shape[1]
        mask = np.zeros(im.shape)
        for bone in ['RH2MC','RH2PP','RH2MP','RH2DP']:
            if os.path.isfile(curveLoc + imName + '_' + bone + '.txt'):
                cv = np.loadtxt(curveLoc + imName + '_' + bone + '.txt')
                cv[:,1] = Nx - cv[:,1]
                pts = cv.astype(np.int32)
                cv2.polylines(mask,[pts],False,(255,255,255),2)

        cropIm   = im[250-17:750+17,1250-17:1550+17]
        cropMask = mask[250:750,1250:1550]/255
        preds = predict_from_im(cropIm,modelLoc)
        preds = preds[17:-17,17:-17]
        plt.figure();plt.imshow(cropIm)
        plt.figure()
        plt.imshow(cropMask)
        plt.figure()
        plt.imshow(1-np.argmax(preds,axis=2))
        score = adjusted_rand_score((cropMask/np.max(cropMask)).ravel(),(1-np.argmax(preds,axis=2)).ravel())

        tf.reset_default_graph()
        patchSize = 17
        modelLoc1 = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/discriminativeModel/trainedModel' \
                   '/16/savedModel'
        #modelName = modelLoc + '/model-5001'
        print('predicting shit')
        #preds = predict_from_imLoc('./validationData/2/True/')
        #preds, names = predict_from_TfRecord('./data/eval-00000-of-00001',modelLoc=modelLoc)
        #coords = np.loadtxt('./validationData/1/coords.txt')
        im = cv2.imread('./validationData/testIm4.png',0)
        #im = 1-cv2.imread('./validationData/CPSA0054h2012.png',0)
        preds1 = predict_from_im(im,modelLoc1,patchSize=patchSize)
        predMax1 = np.argmax(preds1,axis=2)
        #coords = turn_names_to_coords(names)
        #plt.figure()
        #grid_z0 = turn_into_potential(preds[:,0],coords,im,method='cubic')
        ncols = preds1.shape[2]+1
        fig,ax = plt.subplots(nrows=1,ncols=ncols)
        ax[0].imshow(im)
        ax[0].set_title('image')
        for i in range(1,ncols):
            ax[i].imshow(preds1[:,:,i-1])
            ax[i].set_title('class '+ str(i)+ ' probs')
#        fig.savefig(os.path.join(modelLoc1[:-10],'testIm4.png'))
        tf.reset_default_graph()
        modelLoc2 = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/discriminativeModel/trainedModel' \
                   '/18/savedModel'
        #modelName = modelLoc + '/model-5001'
        print('predicting shit')
        #preds = predict_from_imLoc('./validationData/2/True/')
        #preds, names = predict_from_TfRecord('./data/eval-00000-of-00001',modelLoc=modelLoc)
        #coords = np.loadtxt('./validationData/1/coords.txt')
        im = cv2.imread('./validationData/testIm4.png',0)
        #im = 1-cv2.imread('./validationData/CPSA0054h2012.png',0)
        preds2 = predict_from_im(im,modelLoc2,patchSize=patchSize)
        predMax2 = np.argmax(preds2,axis=2)
        #coords = turn_names_to_coords(names)
        #plt.figure()
        #grid_z0 = turn_into_potential(preds[:,0],coords,im,method='cubic')
        ncols = preds2.shape[2]+1
        fig,ax = plt.subplots(nrows=1,ncols=ncols)
        ax[0].imshow(im)
        ax[0].set_title('image')
        for i in range(1,ncols):
            ax[i].imshow(preds2[:,:,i-1])
            ax[i].set_title('class '+ str(i)+ ' probs')
        #fig.savefig(os.path.join(modelLoc2[:-10],'testIm4.png'))


        mergedPred = (preds1[:,:,0]+preds2[:,:,0])/2
        plt.figure()
        plt.imshow(mergedPred)
        plt.title('merged predictions')
        #"""
