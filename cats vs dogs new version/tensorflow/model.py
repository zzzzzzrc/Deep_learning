import tensorflow as tf
def inference(images,batch_size,n_classes):
    #  input images 4D tensor (batch_size,width,height,channels)  tf.float32
    #        batch_size int32
    #        n_classes  int32
    #  ALex input 227*227*3
    # conv1
    with tf.variable_scope('conv1') as scope:
        weight = tf.get_variable('weight1',              # conv1  kernal size 11*11*96
                                 shape=[11,11,3,96],     # initialize means:0 stddev:0.01
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias1',
                               shape=[96],               #bias 96
                               dtype=tf.float32,         #initialize 0
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images,weight,strides=[1,4,4,1],padding='VALID')    #conv
        pre_activation = tf.nn.bias_add(conv,bias)                              # add bias
        conv1 = tf.nn.relu(pre_activation,name=scope.name)                      #relu activation
    #pool1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1,                    #pool1  maxpool  kernal size 3*3
                               ksize=[1,3,3,1],              #stride 2   pad 0
                               strides=[1,2,2,1],
                               padding='VALID',
                               name='pool1')
    #norm1
    with tf.variable_scope('norm1') as scope:                    #norm1 : LRN
        norm1 = tf.nn.lrn(pool1,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.01/9.0,
                          beta=0.75,
                          name='norm1')
    #conv2
    with tf.variable_scope('conv2') as scope:                        #conv2 5*5*256
        weight=tf.get_variable('weight2',                   #stride 1  padding 2
                               shape=[5,5,96,256],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias2',                     #bias 256   init 1
                               shape=[256],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(norm1,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv2 = tf.nn.relu(pre_activation,name=scope.name)
    #pool2
    with tf.variable_scope('pool2') as scope:                     #pool2 ksize : 3*3   stride: 2   padding :0
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1,3,3,1],
                               strides=[1,2,2,1],
                               padding='VALID',
                               name='pool2')
    #norm2
    with tf.variable_scope('norm2') as scope:                    #norm2:LRN
        norm2 = tf.nn.lrn(pool2,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.01/9.0,
                          beta=0.75,
                          name='norm2')
    #conv3
    with tf.variable_scope('conv3') as scope:
        weight = tf.get_variable('weight3',                 #conv3:   size:3*3*384  stride:1  pad:1
                                 shape=[3,3,256,384],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias3',                         #bias3:   size:384  init 0
                               shape=[384],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(norm2,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv3 = tf.nn.relu(pre_activation)
    #conv4
    with tf.variable_scope('conv4') as scope:
        weight=tf.get_variable('weight4',             #conv4  size:3*3*384  stride:1 pad:1
                               shape=[3,3,384,384],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias4',
                               shape=[384],             #bias4  size 384  init 1
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(conv3,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv4 = tf.nn.relu(pre_activation)
    #conv5
    with tf.variable_scope('conv5') as scope:
        weight = tf.get_variable('weight5',          #conv5   3*3*256  stride 1 pad 1
                                 shape=[3,3,384,256],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias5',
                               shape=[256],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(conv4,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv5 = tf.nn.relu(pre_activation)
    #pool5
    with tf.variable_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1,3,3,1],
                               strides=[1,1,1,1],
                               padding='VALID',
                               name='pool5')
    #fc6
    with tf.variable_scope('full_connect6') as scope:
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight = tf.get_variable(name='weight6',
                                 shape=[dim,4096],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias6',
                               shape=[4096],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        pre_activation = tf.matmul(reshape,weight)+bias
        pre_dropout = tf.nn.relu(pre_activation,name=scope.name)
        fc6 = tf.nn.dropout(pre_dropout,keep_prob=0.5)
    #fc7
    with tf.variable_scope('full_connect7') as scope:
        weight = tf.get_variable('weight7',
                                 shape=[4096,4096],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias7',
                               shape=[4096],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        pre_activation = tf.matmul(fc6,weight)+bias
        pre_dropout = tf.nn.relu(pre_activation,name=scope.name)
        fc7 = tf.nn.dropout(pre_dropout,keep_prob=0.5)
    #fc8
    with tf.variable_scope('full_connect8') as scope:
        weight = tf.get_variable('weight8',
                                 shape=[4096,n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias8',
                               shape=[n_classes],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        fc8 = tf.add(tf.matmul(fc7,weight),bias,name='fc8')
    return fc8

def losses(logits,labels):         #交叉熵损失函数
    # logits : 2D tensor [batch_size,n_classes]
    # labels : 1D tensor[batch_size]  tf.float32
    with tf.variable_scope('losses') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def training(loss,learninga_rate):
    with tf.variable_scope('training') as scope:
        optimizer = tf.train.AdamOptimizer(learninga_rate)
        gloalstep = tf.Variable(0,trainable=False,name='gloalstep')
        train_op = optimizer.minimize(loss,global_step=gloalstep)
    return train_op

def evaluation(logits, labels):

    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
#%%
"""def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 32],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)

    #pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        mean,val = tf.nn.moments(pool1,axes=[0])
        epsilon = 0.001
        norm1 = tf.nn.batch_normalization(pool1,mean,val,offset=None,scale=None,variance_epsilon=epsilon)

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,32,32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')


    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')

        mean,val = tf.nn.moments(pool2,axes=[0])
        epsilon = 0.001
        norm2 = tf.nn.batch_normalization(pool2,mean,val,offset=None,scale=None,variance_epsilon=epsilon)



    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,32, 64],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name= scope.name)

    #pool3 and norm3
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling3')
        mean,val = tf.nn.moments(pool3,axes=[0])
        epsilon = 0.001
        norm3 = tf.nn.batch_normalization(pool3,mean,val,offset=None,scale=None,variance_epsilon=epsilon)


    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[256, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op
"""
#%%
"""def evaluation(logits, labels):

  with tf.variable_scope('accuracy') as scope:
      #correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.equal(tf.argmax(logits,1),labels)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%"""
