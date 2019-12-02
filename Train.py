# -*- coding: utf-8 -*-
from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.ops import control_flow_ops

path = 'PATH-DATA'
model_path = 'PATH-MODEL'

batch_size = 64
num_classes = 12
data_size = 4096
time_step = 73
rnn_unit = 73
input_size = 896
output_size = 12
lstm_layers = 2

data = data
labels = labels
Num = len(data)
labelset = []
for i in range(Num):
    labelset.append(labels[i][0]-1)
label = np.asarray(labelset, np.int32)

arr = np.arange(Num)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(Num * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def batch_norm(x, is_training):
    epsilon = 1e-3
    phase_train = tf.convert_to_tensor(is_training, dtype=tf.bool)
    axis = list(range(len(x.get_shape()) - 1))
    with tf.variable_scope('batch_norm'):
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name='beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name='gamma', trainable=True, dtype=x.dtype)
        batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var))
                                          )
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    return normed

weights = {
    'out': tf.Variable(tf.random_normal([rnn_unit, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def LSTM(x, weights, biases):
    x = tf.unstack(x, time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

x = tf.placeholder(tf.float32, shape=[None, data_size], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

def inference(input_tensor, train, regularizer):

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

    with tf.name_scope("layer2-pool1"):
        maxpool_size = 3
        stride_size = 2
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 1, maxpool_size, 1],
                                     strides=[1, 1, stride_size, 1],
                                     padding='VALID')

    with tf.variable_scope("layer3-conv2"):
        NumFilter2 = 64
        conv_size2 = 13
        conv2_filter_weights = tf.get_variable("weight", [1, conv_size2, NumFilter1, NumFilter2],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [NumFilter2], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm(conv2, True)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [1, 15, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_norm(conv3, True)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 1, maxpool_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')
        Input_lstm = tf.reshape(pool3, [-1, rnn_unit, input_size])

    with tf.variable_scope("layer9-lstm"):
        logits = LSTM(Input_lstm, weights, biases)
    return logits

regularizer = None
logits = inference(x, False, regularizer)
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

stop = False
last_improvement=0
epoch = 0
max_epochs = 1000
require_improvement= 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Train_Loss = []
    Train_Acc =[]
    Validation_Loss = [100000]
    Validation_Acc = []

    while epoch < max_epochs and stop == False:
        start_time = time.time()

        train_loss, train_acc, n_batch = 0, 0, 0
        Temp_Train_Loss = []
        Temp_Train_Acc = []
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
            print("   train loss: %f" % (np.sum(train_loss) / n_batch))
            print("   train acc: %f" % (np.sum(train_acc) / n_batch))
            Temp_Train_Loss.append( np.sum(train_loss) / n_batch )
            Temp_Train_Acc.append( np.sum(train_acc) / n_batch )

        val_loss, val_acc, n_batch = 0, 0, 0
        Temp_Validation_Loss = []
        Temp_Validation_Acc = []
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
            print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
            print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
            Temp_Validation_Loss.append( np.sum(val_loss) / n_batch )
            Temp_Validation_Acc.append( np.sum(val_acc) / n_batch )

        BestLoss = np.min(Temp_Validation_Loss)
        BaselineLoss = np.min(Validation_Loss)
        if BestLoss < BaselineLoss:
            last_improvement = 0
        else:
            last_improvement += 1
        if last_improvement > require_improvement:
            print("No improvement found during the last iterations, stopping optimization.")
            stop = True

        Train_Loss.append(np.min(Temp_Train_Loss))
        Train_Acc.append( Temp_Train_Acc[Temp_Train_Loss.index(min(Temp_Train_Loss))] )
        Validation_Loss.append(np.min(Temp_Validation_Loss))
        Validation_Acc.append( Temp_Validation_Acc[Temp_Validation_Loss.index(min(Temp_Validation_Loss))] )

        saver.save(sess, model_path, global_step=epoch)
        print(sess.graph.name_scope)
