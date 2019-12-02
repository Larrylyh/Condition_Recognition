import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def D1LeNet5(input_tensor):
    with tf.variable_scope('layer1-conv1'):
        NumFilter1 = 64
        conv_size1 = 11
        conv1_filter_weights = tf.get_variable("weight", [1, conv_size1, 1, NumFilter1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [NumFilter1], initializer=tf.constant_initializer(0.0))
        input_3d = tf.expand_dims(input_tensor, 1)
        input_4d = tf.expand_dims(input_3d, -1)
        convolution_output = tf.nn.conv2d(input_4d, filter=conv1_filter_weights,
                                          strides=[1, 1, 1, 1], padding="SAME")
        conv1 = batch_norm(convolution_output, True)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        print('relu1:', relu1.shape)

    with tf.name_scope("layer2-pool1"):
        maxpool_size = 3
        stride_size = 2
        pool1 = tf.nn.avg_pool(relu1, ksize=[1, 1, maxpool_size, 1],
                                     strides=[1, 1, stride_size, 1],
                                     padding='VALID')
        print('pool1:',pool1.shape)

    with tf.variable_scope("layer3-conv2"):
        NumFilter2 = 128
        conv_size2 = 13
        conv2_filter_weights = tf.get_variable("weight", [1, conv_size2, NumFilter1, NumFilter2],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [NumFilter2], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm(conv2, True)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.avg_pool(relu2, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        neronodes = 1024
        fc1_weights = tf.get_variable("weight", [nodes, neronodes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [neronodes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        neronodes1 = 512
        fc2_weights = tf.get_variable("weight", [neronodes, neronodes1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [neronodes1], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-fc3'):
        nclass = 12
        fc3_weights = tf.get_variable("weight", [neronodes1, nclass],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [nclass], initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, fc3_weights), fc3_biases, name="output")
    return logits


def D1AlexNet(input_tensor):
    with tf.variable_scope('layer1-conv1'):
        NumFilter1 = 32
        conv_size1 = 11
        conv1_filter_weights = tf.get_variable("weight", [1, conv_size1, 1, NumFilter1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [NumFilter1], initializer=tf.constant_initializer(0.0))
        input_3d = tf.expand_dims(input_tensor, 1)
        input_4d = tf.expand_dims(input_3d, -1)
        convolution_output = tf.nn.conv2d(input_4d, filter=conv1_filter_weights,
                                          strides=[1, 1, 1, 1], padding="SAME")
        conv1 = batch_norm(convolution_output, True)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        print('relu1:', relu1.shape)

    with tf.name_scope("layer2-pool1"):
        maxpool_size = 3
        stride_size = 2
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 1, maxpool_size, 1],
                                     strides=[1, 1, stride_size, 1],
                                     padding='VALID')
        print('pool1:',pool1.shape)

    with tf.variable_scope("layer3-conv2"):
        NumFilter2 = 64
        conv_size2 = 5
        conv2_filter_weights = tf.get_variable("weight", [1, conv_size2, NumFilter1, NumFilter2],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [NumFilter2], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm(conv2, True)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        NumFilter3 = 128
        conv_size3 = 3
        conv3_filter_weights = tf.get_variable("weight", [1, conv_size3, NumFilter2, NumFilter3],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [NumFilter3], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_norm(conv3, True)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope("layer6-conv4"):
        NumFilter4 = 128
        conv_size4 = 3
        conv4_filter_weights = tf.get_variable("weight", [1, conv_size4, NumFilter3, NumFilter4],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [NumFilter4], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = batch_norm(conv4, True)
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.variable_scope("layer7-conv5"):
        NumFilter5 = 128
        conv_size5 = 3
        conv5_filter_weights = tf.get_variable("weight", [1, conv_size5, NumFilter4, NumFilter5],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [NumFilter5], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = batch_norm(conv5, True)
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.name_scope("layer8-pool3"):
        pool3 = tf.nn.max_pool(relu5, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1],
                               padding='VALID')
        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        neronodes = 1024
        fc1_weights = tf.get_variable("weight", [nodes, neronodes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [neronodes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        neronodes1 = 512
        fc2_weights = tf.get_variable("weight", [neronodes, neronodes1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [neronodes1], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        nclass = 12
        fc3_weights = tf.get_variable("weight", [neronodes1, nclass],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [nclass], initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, fc3_weights), fc3_biases, name="output")
    return logits


def D1VGG16(input_tensor):
    with tf.variable_scope('layer1-conv1'):
        NumFilter1 = 16
        conv_size1 = 3
        conv1_filter_weights = tf.get_variable("weight", [1, conv_size1, 1, NumFilter1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [NumFilter1], initializer=tf.constant_initializer(0.0))
        input_3d = tf.expand_dims(input_tensor, 1)
        input_4d = tf.expand_dims(input_3d, -1)
        convolution_output = tf.nn.conv2d(input_4d, filter=conv1_filter_weights,
                                          strides=[1, 1, 1, 1], padding="SAME")
        conv1 = batch_norm(convolution_output, True)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        print('relu1:', relu1.shape)

    with tf.variable_scope("layer2-conv2"):
        NumFilter2 = 16
        conv_size2 = 3
        conv2_filter_weights = tf.get_variable("weight", [1, conv_size2, NumFilter1, NumFilter2],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [NumFilter2], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm(conv2, True)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer3-pool1"):
        maxpool_size = 3
        stride_size = 2
        pool1 = tf.nn.max_pool(relu2, ksize=[1, 1, maxpool_size, 1],
                                     strides=[1, 1, stride_size, 1],
                                     padding='VALID')
        print('pool1:',pool1.shape)

    with tf.variable_scope("layer4-conv3"):
        NumFilter3 = 32
        conv_size3 = 3
        conv3_filter_weights = tf.get_variable("weight", [1, conv_size3, NumFilter2, NumFilter3],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [NumFilter3], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1, conv3_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_norm(conv3, True)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope("layer5-conv4"):
        NumFilter4 = 32
        conv_size4 = 3
        conv4_filter_weights = tf.get_variable("weight", [1, conv_size4, NumFilter3, NumFilter4],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [NumFilter4], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = batch_norm(conv4, True)
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')

    with tf.variable_scope("layer7-conv5"):
        NumFilter5 = 64
        conv_size5 = 3
        conv5_filter_weights = tf.get_variable("weight", [1, conv_size5, NumFilter4, NumFilter5],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [NumFilter5], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2, conv5_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = batch_norm(conv5, True)
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.variable_scope("layer8-conv6"):
        NumFilter6 = 64
        conv_size6 = 3
        conv6_filter_weights = tf.get_variable("weight", [1, conv_size6, NumFilter5, NumFilter6],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [NumFilter6], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = batch_norm(conv6, True)
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

    with tf.variable_scope("layer9-conv7"):
        NumFilter7 = 64
        conv_size7 = 3
        conv7_filter_weights = tf.get_variable("weight", [1, conv_size7, NumFilter6, NumFilter7],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable("bias", [NumFilter7], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(relu6, conv7_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv7 = batch_norm(conv7, True)
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))

    with tf.name_scope("layer10-pool3"):
        pool3 = tf.nn.max_pool(relu7, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1],
                               padding='VALID')

    with tf.variable_scope("layer11-conv8"):
        NumFilter8 = 128
        conv_size8 = 3
        conv8_filter_weights = tf.get_variable("weight", [1, conv_size8, NumFilter7, NumFilter8],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv8_biases = tf.get_variable("bias", [NumFilter8], initializer=tf.constant_initializer(0.0))
        conv8 = tf.nn.conv2d(pool3, conv8_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv8 = batch_norm(conv8, True)
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases))

    with tf.variable_scope("layer12-conv9"):
        NumFilter9 = 128
        conv_size9 = 3
        conv9_filter_weights = tf.get_variable("weight", [1, conv_size9, NumFilter8, NumFilter9],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv9_biases = tf.get_variable("bias", [NumFilter9], initializer=tf.constant_initializer(0.0))
        conv9 = tf.nn.conv2d(relu8, conv9_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv9 = batch_norm(conv9, True)
        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))

    with tf.variable_scope("layer13-conv10"):
        NumFilter10 = 128
        conv_size10 = 3
        conv10_filter_weights = tf.get_variable("weight", [1, conv_size10, NumFilter9, NumFilter10],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv10_biases = tf.get_variable("bias", [NumFilter10], initializer=tf.constant_initializer(0.0))
        conv10 = tf.nn.conv2d(relu9, conv10_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv10 = batch_norm(conv10, True)
        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))

    with tf.name_scope("layer14-pool4"):
        pool4 = tf.nn.max_pool(relu10, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1],
                               padding='VALID')

    with tf.variable_scope("layer15-conv11"):
        NumFilter11 = 256
        conv_size11 = 3
        conv11_filter_weights = tf.get_variable("weight", [1, conv_size11, NumFilter10, NumFilter11],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv11_biases = tf.get_variable("bias", [NumFilter11], initializer=tf.constant_initializer(0.0))
        conv11 = tf.nn.conv2d(pool4, conv11_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv11 = batch_norm(conv11, True)
        relu11 = tf.nn.relu(tf.nn.bias_add(conv11, conv11_biases))

    with tf.variable_scope("layer16-conv12"):
        NumFilter12 = 256
        conv_size12 = 3
        conv12_filter_weights = tf.get_variable("weight", [1, conv_size12, NumFilter11, NumFilter12],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv12_biases = tf.get_variable("bias", [NumFilter12], initializer=tf.constant_initializer(0.0))
        conv12 = tf.nn.conv2d(relu11, conv12_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv12 = batch_norm(conv12, True)
        relu12 = tf.nn.relu(tf.nn.bias_add(conv12, conv12_biases))

    with tf.variable_scope("layer17-conv13"):
        NumFilter13 = 256
        conv_size13 = 3
        conv13_filter_weights = tf.get_variable("weight", [1, conv_size13, NumFilter12, NumFilter13],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv13_biases = tf.get_variable("bias", [NumFilter13], initializer=tf.constant_initializer(0.0))
        conv13 = tf.nn.conv2d(relu12, conv13_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv13 = batch_norm(conv13, True)
        relu13 = tf.nn.relu(tf.nn.bias_add(conv13, conv13_biases))

    with tf.name_scope("layer18-pool5"):
        pool5 = tf.nn.max_pool(relu13, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1],
                               padding='VALID')
        pool_shape = pool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool5, [-1, nodes])

    with tf.variable_scope('layer19-fc1'):
        neronodes = 1024
        fc1_weights = tf.get_variable("weight", [nodes, neronodes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [neronodes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer20-fc2'):
        neronodes1 = 512
        fc2_weights = tf.get_variable("weight", [neronodes, neronodes1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [neronodes1], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        nclass = 12
        fc3_weights = tf.get_variable("weight", [neronodes1, nclass],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [nclass], initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, fc3_weights), fc3_biases, name="output")
    return logits

def LSTM(input_tensor):
    input_3d = tf.expand_dims(input_tensor, 1)
    Input_lstm = tf.reshape(input_3d, [-1, rnn_unit, input_size])
    x = tf.unstack(Input_lstm, time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return logits

def MLP(input_tensor):
    input_3d = tf.expand_dims(input_tensor, 1)
    Input_lstm = tf.reshape(input_3d, [-1, rnn_unit, input_size])
    x = tf.unstack(Input_lstm, time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return logits
