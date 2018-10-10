import tensorflow as tf
from utils import del_all_flags

#del_all_flags(tf.flags.FLAGS)
#"""
tf.app.flags.DEFINE_integer('kernel_size_1',5,
                           'Size of first convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_2',5,
                           'Size of second convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_3',3,
                           'Size of third convolutional kernel')
tf.app.flags.DEFINE_integer('kernel_size_4',3,
                           'Size of fourth convolutional kernel')
FLAGS = tf.app.flags.FLAGS
#"""


def conv_layer(inputs
              ,filters
              ,kernel_size
              ,padding
              ,activation
              ,name
              ):
    with tf.name_scope(name):
        output = tf.layers.conv2d(
                    inputs=inputs
                    ,filters=filters
                    ,kernel_size=kernel_size
                    ,padding=padding
                    ,activation=activation
                )
        tf.summary.histogram('conv',output)
    return output

def pool_layer(inputs,pool_size,strides,padding,name,pool_type='max'):
    with tf.name_scope(name):
        if pool_type=='max' or pool_type=='MAX':
            output = tf.layers.average_pooling2d(inputs=inputs
                                          ,pool_size=pool_size
                                          ,strides=strides
                                          ,padding=padding)
        else:
            output = tf.layers.average_pooling2d(inputs=inputs
                                          ,pool_size=pool_size
                                          ,strides=strides
                                          ,padding=padding)
        tf.summary.histogram('pools',output)
    return output

def full_layer(inputs,units,activation,name):
    with tf.name_scope(name):
        output = tf.layers.dense(inputs=inputs
                                 ,units=units
                                 ,activation=activation
                                 ,name=name)
        tf.summary.histogram('dropout',output)
    return output


"""
CONV_RELU_48-MAX_POOL\n\
CONV_RELU_64-MAX_POOL\n\
CONV_RELU_128-MAX_POOL\n\
FC_SOFTMAX_128\n\
FL_SOFTMAX_nCLASS
"""
def adnet_1(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3')

    kx = int(imSize[0]/2**3)
    ky = int(imSize[1]/2**3)
    nets = conv_layer(nets,filters=70,kernel_size=[kx,ky],padding='valid',activation=None,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets =  tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets


"""
CONV_RELU_32-CONV_RELU_32-MAX_POOL\n\
CONV_RELU_64-CONV_RELU_64-MAX_POOL\n\
CONV_RELU_128-CONV_RELU_128-CONV_RELU_128-MAX_POOL\n\
FC_SOFTMAX_128\n\
FL_SOFTMAX_nCLASS
"""

def adnet_2(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [23,23]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    #conv layers
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3')

    #fc uses conv koz its easier to implement
    kx = int(imSize[0]/2**3)
    ky = int(imSize[1]/2**3)
    nets = conv_layer(nets,filters=128,kernel_size=[kx,ky],padding='valid',activation=tf.nn.relu,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets =  tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets



"""
CONV_RELU_20-MAX_POOL
CONV_RELU_40-MAX_POOL
CONV_RELU_80-MAX_POOL
FC_SOFTMAX_80
FL_SOFTMAX_nCLASS
"""
def adnet_3(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3')

    kx = int(imSize[0]/2**3)
    ky = int(imSize[1]/2**3)
    nets = conv_layer(nets,filters=80,kernel_size=[kx,ky],padding='valid',activation=tf.nn.relu,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets =  tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets



"""
CONV_RELU_32-CONV_RELU_32-MAX_POOL
CONV_RELU_64-CONV_RELU_64-MAX_POOL
FC_SOFTMAX_64
FL_SOFTMAX_nCLASS
"""
def adnet_4(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')

    kx = int(imSize[0]/2**2)
    ky = int(imSize[1]/2**2)
    nets = conv_layer(nets,filters=64,kernel_size=[kx,ky],padding='valid',activation=tf.nn.softmax,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets = tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets




"""
CONV_RELU_32-MAX_POOL
CONV_RELU_64-MAX_POOL
FC_SOFTMAX_128
FL_SOFTMAX_nCLASS
"""
def adnet_5(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')

    kx = int(imSize[0]/2**2)
    ky = int(imSize[1]/2**2)
    nets = conv_layer(nets,filters=80,kernel_size=[kx,ky],padding='valid',activation=tf.nn.softmax,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets = tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets



"""
CONV_SPLUS_32-MAX_POOL
CONV_SPLUS_64-MAX_POOL
CONV_SPLUS_128-MAX_POOL
CONV_SPLUS_256-MAX_POOL
FC_SPLUS_256
FL_SPLUS_nCLASS
"""
def adnet_6(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    kernel_size4 = FLAGS.kernel_size_4
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.softplus,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.softplus,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.softplus,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=256,kernel_size=kernel_size4,padding='same',activation=tf.nn.softplus,name='conv4')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv4')


    kx = int(imSize[0]/2**4)
    ky = int(imSize[1]/2**4)
    nets = conv_layer(nets,filters=256,kernel_size=[kx,ky],padding='valid',activation=tf.nn.softplus,name = 'fc1')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets = tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets


"""
CONV_RELU_32-MAX_POOL
CONV_RELU_64-MAX_POOL
CONV_RELU_128-MAX_POOL
CONV_RELU_256-MAX_POOL
FC_RELU_256
FL_SOFTMAX_nCLASS
"""
def adnet_7(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [17,17]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    kernel_size4 = FLAGS.kernel_size_4
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=256,kernel_size=kernel_size4,padding='same',activation=tf.nn.relu,name='conv4')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv4')


    kx = int(imSize[0]/2**4)
    ky = int(imSize[1]/2**4)
    nets = conv_layer(nets,filters=256,kernel_size=[kx,ky],padding='valid',activation=tf.nn.relu,name = 'fc1')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets = tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets


"""
CONV_RELU_32-CONV_RELU_32-AVG_POOL
CONV_RELU_64-CONV_RELU_64-AVG_POOL
CONV_RELU_128-CONV_RELU_128-CONV_RELU_128-AVG_POOL
FC_SOFTMAX_128
FL_SOFTMAX_nCLASS
"""

def adnet_8(inputs
           ,num_classes=2
           ,is_training=True
           ,drop_prob=0.5
           ,imSize = [23,23]
           ,batch_norm = False
           ):
    kernel_size1 = FLAGS.kernel_size_1
    kernel_size2 = FLAGS.kernel_size_2
    kernel_size3 = FLAGS.kernel_size_3
    nets =  tf.reshape(inputs,[-1,imSize[0],imSize[1],1],name='Reshape_op')
    #conv layers
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = conv_layer(nets,filters=32,kernel_size=kernel_size1,padding='same',activation=tf.nn.relu,name='conv1')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv1',pool_type='avg')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = conv_layer(nets,filters=64,kernel_size=kernel_size2,padding='same',activation=tf.nn.relu,name='conv2')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv2',pool_type='avg')
    if batch_norm:
        nets = tf.layers.batch_normalization(nets,training=is_training)
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = conv_layer(nets,filters=128,kernel_size=kernel_size3,padding='same',activation=tf.nn.relu,name='conv3')
    nets = pool_layer(nets,pool_size=[2,2],strides=2,padding='valid',name='conv3',pool_type='avg')

    #fc uses conv koz its easier to implement
    kx = int(imSize[0]/2**3)
    ky = int(imSize[1]/2**3)
    nets = conv_layer(nets,filters=128,kernel_size=[kx,ky],padding='valid',activation=tf.nn.relu,name = 'conv')
    #nets = full_layer(nets,units=128,activation=tf.nn.relu,name='FC1')
    nets =  tf.layers.dropout(nets,rate=drop_prob,training=is_training)
    nets = full_layer(nets,units=num_classes,activation=None,name='logits')
    nets = tf.reshape(nets,[-1,num_classes])
    return nets




model_library = {
     '1' :adnet_1
    ,'2' :adnet_2
    ,'3' :adnet_3
    ,'4' :adnet_4
    ,'5' :adnet_5
    ,'6' :adnet_6
    ,'7' :adnet_7
    ,'8' :adnet_8
}

architecture_library = {
    '1' : 'CONV_RELU_48-MAX_POOL\n\
CONV_RELU_64-MAX_POOL\n\
CONV_RELU_128-MAX_POOL\n\
FC_SOFTMAX_128'

    ,'2': 'CONV_RELU_32-CONV_RELU_32-MAX_POOL\n\
CONV_RELU_64-CONV_RELU_64-MAX_POOL\n\
CONV_RELU_128-CONV_RELU_128-CONV_RELU_128-MAX_POOL\n\
FC_SOFTMAX_128\n\
FL_SOFTMAX_nCLASS'

    ,'3': 'CONV_RELU_20-MAX_POOL\n\
CONV_RELU_40-MAX_POOL\n\
CONV_RELU_80-MAX_POOL\n\
FC_SOFTMAX_80\n\
FL_SOFTMAX_nCLASS'

    ,'4': 'CONV_RELU_32-CONV_RELU_32-MAX_POOL\n\
CONV_RELU_64-CONV_RELU_64-MAX_POOL\n\
FC_SOFTMAX_64\n\
FL_SOFTMAX_nCLASS'

    ,'5':'CONV_RELU_32-MAX_POOL\n\
CONV_RELU_64-MAX_POOL\n\
FC_SOFTMAX_128\n\
FL_SOFTMAX_nCLASS'

    ,'6':'CONV_SPLUS_32-MAX_POOL\n\
CONV_SPLUS_64-MAX_POOL\n\
CONV_SPLUS_128-MAX_POOL\n\
CONV_SPLUS_256-MAX_POOL\n\
FC_SPLUS_256\n\
FL_SPLUS_nCLASS'


    ,'7': 'CONV_RELU_32-MAX_POOL\n\
CONV_RELU_64-MAX_POOL\n\
CONV_RELU_128-MAX_POOL\n\
CONV_RELU_256-MAX_POOL\n\
FC_RELU_256\n\
FL_SOFTMAX_nCLASS'

    ,'8': 'CONV_RELU_32-CONV_RELU_32-AVG_POOL\n\
CONV_RELU_64-CONV_RELU_64-AVG_POOL\n\
CONV_RELU_128-CONV_RELU_128-CONV_RELU_128-AVG_POOL\n\
FC_SOFTMAX_128\n\
FL_SOFTMAX_nCLASS'
}
