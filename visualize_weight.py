import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import models
def visualize_weight(ckptfile, graph) :
    with tf.Session(graph = graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, ckptfile)

        w1 = graph.get_tensor_by_name("w1:0")
        result = sess.run(w1)
        p1 = np.reshape(result[:,:,:,10], [5,5,4])
        print(p1.shape)
        plt.imshow(p1)

cnn_g , cnn_op = models.cnn_graph()

visualize_weight("./ckpt/cnn4.ckpt", cnn_g)
