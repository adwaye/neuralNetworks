import numpy as np
import tensorflow as tf
import os
import cv2
slim = tf.contrib.slim
import tensorflow.contrib.slim.nets as nets
import sys
sys.path.append('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/discriminativeModel/')
#from checkTfrecords import check_size,count_data
#from tensorflow.contrib.framework import assign_from_checkpoint_fn
import scipy.io as sio
from utils import update_dir
import matplotlib.pyplot as plt
from tensorflow.contrib.opt import ScipyOptimizerInterface
plt.ion()

def _eval_tensor(x):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        output = sess.run(x)

    return output



vgg = nets.vgg



height = 224
width  = 224
dType = tf.float32


matFile    = sio.loadmat('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/artStuff/vgg_16_conv.mat')
convList   = matFile['convList']
variables  = matFile['weights']
names      = matFile['names']



#                     1_1  1_2  2_1  2_2  3_1  3_2  3_3  4_1  4_2  4_3  5_1  5_2  5_3
weightSt  = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
weightSt  = weightSt/np.sum(weightSt)
weightCon = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
weightCon = weightCon/np.sum(weightCon)
def transferArtStyle(artImage,Image,alpha = 1,beta  = 0.001,regWeight=0.01,
                     weightStyle   = weightSt,
                     weightContent = weightCon,lrate = 0.01,momentum=0.001,Niter = 2000):
    tf.reset_default_graph()
    outPutIm  = np.random.random(Image.shape)
    #outPutIm  = np.ones(Image.shape)
    #outPutIm = outPutIm + np.min(outPutIm)

    with tf.name_scope('image_input'):
        tf_style_image   = tf.constant(artImage , dtype=dType, name = 'style_image')
        tf_content_image = tf.constant(Image , dtype=dType, name = 'content_image')
        tf_synth_image    = tf.Variable(outPutIm, dtype = dType, name = 'synthesized_image')

    #convolve and compute gram matrices and shit
    conv_list_style   = []
    conv_list_synth   = []
    conv_list_content = []
    gram_list_style   = []
    gram_list_synth   = []



    with tf.name_scope('filtered_images'):
        layer = 1
        k = 0
        for conv in matFile['convList']:
            if int(conv[11])>layer:
                layer += 1
                convolved_style = tf.nn.max_pool(convolved_style
                                           ,ksize=[1,2,2,1]
                                           ,strides=[1,2,2,1]
                                           ,padding='VALID'
                                           ,data_format='NHWC'
                                           )
                convolved_synth = tf.nn.max_pool(convolved_synth
                                           ,ksize=[1,2,2,1]
                                           ,strides=[1,2,2,1]
                                           ,padding='VALID'
                                           ,data_format='NHWC'
                                                 )
                convolved_cont  = tf.nn.max_pool(convolved_cont
                                           ,ksize=[1,2,2,1]
                                           ,strides=[1,2,2,1]
                                           ,padding='VALID'
                                           ,data_format='NHWC'
                                                 )

            filter = tf.constant(variables[0][np.where(names == conv + '/weights:0')[0][0]],dtype=dType)
            bias = tf.constant(
                np.expand_dims(
                    np.expand_dims(variables[0][np.where(names == conv + '/biases:0 ')[
                        0][0]]
                                   ,0)
                    ,0)
                ,)
            if k ==0:
                convolved_style = tf.nn.conv2d(tf_style_image,filter=filter,strides =[1,1,1,1], padding = 'SAME')
                convolved_synth = tf.nn.conv2d(tf_synth_image,filter=filter,strides =[1,1,1,1], padding = 'SAME')


                convolved_cont  = tf.nn.conv2d(tf_content_image,filter=filter,strides =[1,1,1,1], padding = 'SAME')
            else:
                convolved_style = tf.nn.conv2d(convolved_style,filter=filter,strides =[1,1,1,1], padding = 'SAME')
                convolved_synth = tf.nn.conv2d(convolved_synth,filter=filter,strides =[1,1,1,1], padding = 'SAME')
                convolved_cont  = tf.nn.conv2d(convolved_cont,filter=filter,strides =[1,1,1,1], padding = 'SAME')


            convolved_style = tf.nn.relu(tf.add(convolved_style,bias),name=conv)
            convolved_synth = tf.nn.relu(tf.add(convolved_synth,bias),name=conv)
            convolved_cont  = tf.nn.relu(tf.add(convolved_cont,bias),name=conv)
            Ml = convolved_style.shape[1].value * convolved_style.shape[2].value
            Nl = convolved_style.shape[3].value

            convolved_style_flat = tf.reshape(convolved_style,[Ml,Nl])
            gram_style = tf.matmul(
                    tf.transpose(convolved_style_flat),
                    convolved_style_flat
                )

            convolved_synth_flat = tf.reshape(convolved_synth,[Ml,Nl])
            gram_synth = tf.matmul(
                    tf.transpose(convolved_synth_flat),
                    convolved_synth_flat
                )
            if k == 0:
                styleLoss   = (weightStyle[k]/(4*(Ml*Nl)**2))*tf.reduce_sum(tf.squared_difference(gram_synth,
                                                                                                  gram_style))
                contentLoss = 0.5*weightContent[k]* tf.reduce_sum(tf.squared_difference(convolved_synth,convolved_cont))
            else:
                styleLoss = styleLoss + tf.scalar_mul(weightStyle[k] / (4 * (Ml * Nl) ** 2),tf.reduce_sum(
                    tf.squared_difference(gram_synth,gram_style))   )
                contentLoss = contentLoss+ tf.scalar_mul(0.5*weightContent[k],tf.reduce_sum(tf.squared_difference(
                    convolved_synth,convolved_cont))  )


            conv_list_style   = conv_list_style +[convolved_style]
            conv_list_synth   = conv_list_synth +[convolved_synth]
            conv_list_content = conv_list_content +[convolved_cont]
            gram_list_style   = gram_list_style+[gram_style]
            gram_list_synth   = gram_list_synth+[gram_synth]
            k += 1

    synth_roll_r = tf.manip.roll(tf_synth_image,shift=1,axis=0)
    synth_roll_l = tf.manip.roll(tf_synth_image,shift=-1,axis=0)
    synth_roll_u = tf.manip.roll(tf_synth_image,shift=1,axis=1)
    synth_roll_d = tf.manip.roll(tf_synth_image,shift=-1,axis=1)

    grad_x  = tf.squared_difference(synth_roll_r,synth_roll_l)
    grad_y  = tf.squared_difference(synth_roll_u,synth_roll_d)
    regLoss = tf.reduce_sum(grad_x)+tf.reduce_sum(grad_y)

    total_loss = contentLoss*alpha + styleLoss*beta + regLoss*regWeight
    #total_loss =  styleLoss

    #optim      = tf.train.GradientDescentOptimizer(lrate)
    optim = tf.train.AdamOptimizer(learning_rate = 1e1,beta1 = 0.9,beta2 = 0.999,epsilon= 1e-08)
    train_step = optim.minimize(total_loss,global_step=tf.train.get_global_step())

    optimizer = ScipyOptimizerInterface(total_loss, method='bfgs')



    fig, axes = plt.subplots(nrows=1,ncols=3)
    axes[0].imshow(xRayImage[0,:,:,:],cmap='gray')
    axes[1].imshow(artImage[0,:,:,:])
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        #with tf.device('gpu:0'):
        for i in range(Niter):
                sess.run(train_step)
                #optimizer.minimize(session=sess)
                if i%30==0:
                    print(i)
                    im = sess.run(tf_synth_image)
                    axes[2].imshow(im[0,:,:,:])
                    fig.suptitle('iteration='+str(i)+' alpha='+str(alpha)+' beta='+str(beta)+' lrate='+str(lrate)+' '
                                                                                                                  'REG='+str(regWeight))
                    plt.pause(0.01)

        im = sess.run(tf_synth_image)
    return im


if __name__=='__main__':
    artLoc   = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/artStuff/artPics'
    artNames = os.listdir(artLoc)
    picLoc   = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/artStuff/pics'
    picName  = 'sexiboii.jpg'
    #or art in artNames:
    art = 'picasso.jpg'
    artImage = cv2.imread(os.path.join(artLoc,art))
    artImage = cv2.resize(artImage,(height,width),interpolation=cv2.INTER_CUBIC)
    artImage = np.expand_dims(artImage,0).astype(np.float32) / (np.max(artImage))

    xRayImage = cv2.imread(os.path.join(picLoc,picName))
    xRayImage = cv2.resize(xRayImage,(height,width),interpolation=cv2.INTER_CUBIC)
    xRayImage = np.expand_dims(xRayImage,0).astype(np.float32) / (np.max(xRayImage))
    alpha = 0.00001
    beta  = 1
    regWeight = 0.01
    lrate = 0.01
    momentum = 0.001
    Niter = 4000
    #                    1_1 1_2 2_1 2_2 3_1 3_2 3_3 4_1 4_2 4_3 5_1 5_2 5_3
    weightSt = np.array([1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    weightSt = weightSt / np.sum(weightSt)
    #                     1_1 1_2 2_1 2_2 3_1 3_2 3_3 4_1 4_2 4_3 5_1 5_2 5_3
    weightCon = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    weightCon = weightCon / np.sum(weightCon)


    im = transferArtStyle(artImage,xRayImage,alpha=alpha,beta=beta,regWeight=regWeight,lrate=lrate,momentum=momentum,\
                                                                                                    Niter=Niter)
    im = im + np.min(im)
    #
    # im = (im/np.max(im)+255).astype(np.uint8)
    output_dir = update_dir('./results')
    fileName = '_a'+str(alpha)+'_b'+str(beta)+'_N'+str(Niter)+art[:-4]+picName[:-4]+'.png'
    plt.savefig(os.path.join(output_dir,fileName))
    np.savetxt(os.path.join(output_dir,'weightStyle.txt'),weightSt)
    np.savetxt(os.path.join(output_dir,'weightCon.txt'),weightCon)

    x = np.array([[1,2,3,4,5],
                  [6,7,8,9,10],
                  [11,12,13,14,15],
                  [16,17,18,19,20],
                  [21,22,23,24,25]
                  ])
    x = np.expand_dims(x,0)
    x = np.expand_dims(x,3)
    x = np.repeat(x,10,axis=3)
    t_x = tf.constant(x,dtype=dType)
    t_x_flat = tf.reshape(t_x,[x.shape[1]*x.shape[2],x.shape[3]])


