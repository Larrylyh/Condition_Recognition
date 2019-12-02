import tensorflow as tf
import numpy as np
import pandas as pd
import os  # os 处理文件和目录的模块
import glob  # glob 文件通配符模块

path = 'PATH-TEST-DATA'
path_meta = 'PATH-meta'
path_ck = 'PATH-ck'
data, labels = read_data(path)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path_meta)
    saver.restore(sess, tf.train.latest_checkpoint(path_ck))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}
    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits, feed_dict)
    print(classification_result)
    predictions = tf.argmax(classification_result, 1).eval()
    print(predictions)
    confuse_martix = sess.run(tf.convert_to_tensor(tf.confusion_matrix(predictions, labels)))
    print(confuse_martix)