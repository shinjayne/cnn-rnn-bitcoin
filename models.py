import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def cnn_graph(learning_rate = 0.001):
    cnn = tf.Graph()
    with cnn.as_default() :

        #feed
        X = tf.placeholder(name = "X", shape = [None, 50, 50 , 4], dtype = np.float32)
        Y = tf.placeholder(name = "Y", shape = [None, 1], dtype = np.float32)

        #TRAINABLE variables
        w1 = tf.get_variable(shape = [5,5,4,64], initializer = tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "w1")
        w1_2 = tf.get_variable(shape = [7,7,64,32], initializer = tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "w1_2")
        w2 = tf.get_variable(shape = [3,3,32,64], initializer= tf.random_normal_initializer(stddev = 0.01, mean= 0.0), name = 'w2')

        #w3 = tf.get_variable(shape = [13*13*64 , 256], initializer= tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "w3")
        w3 = tf.get_variable(shape = [25*25*32 , 256], initializer= tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "w3")
        w4 = tf.get_variable(shape = [256 , 1], initializer= tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "w4")

        b3 = tf.get_variable(shape = [256], initializer= tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "b3")
        b4 = tf.get_variable(shape = [1], initializer= tf.random_normal_initializer(stddev = 0.01, mean = 0.0), name = "b4")

        #Operations

        #<conv1>
        conv1_c = tf.nn.conv2d(X, w1 , strides = [1,1,1,1] , padding = 'SAME')
        #conv1_c shape : [None, 50,50,32]
        conv1_r = tf.nn.relu(conv1_c)

        conv1_2_c = tf.nn.conv2d(conv1_r, w1_2 , strides = [1,1,1,1] , padding = 'SAME')
        conv1_2_r = tf.nn.relu(conv1_2_c)

        conv1_p = tf.nn.max_pool(conv1_2_r, ksize = [1,2,2,1] , strides = [1,2,2,1], padding = 'SAME')
        #conv1_p shape : [None, 25,25, 32]

        ##<conv2>
        #conv2_c = tf.nn.conv2d(conv1_p , w2, strides = [1,1,1,1], padding = 'SAME')
        ##conv2_c shape : [None, 25, 25, 64]
        #conv2_r = tf.nn.relu(conv2_c)
        #conv2_p = tf.nn.max_pool(conv2_r, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        ##conv2_p shape : [None, 13, 13 ,64]

        #<flattening>
        #flattened = tf.reshape(conv2_p , [-1, 13*13*64])
        flattened = tf.reshape(conv1_p , [-1, 25*25*32])

        #<2 layer fully connected NN>
        fc_1 = tf.nn.relu(tf.matmul(flattened , w3) + b3)
        fc_2 = tf.nn.sigmoid(tf.matmul(fc_1, w4) + b4)


        #Loss function : square mean
        loss = tf.reduce_mean(tf.square(fc_2 - Y ))

        #Optimizer : Adam Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        #Accuracy
        fc_2 = fc_2 > 0.5
        fc_2 = tf.to_float(fc_2)

        accuracy = tf.reduce_mean(tf.to_float(tf.equal(fc_2, Y)))
    tf.reset_default_graph()
    op_list = [X, Y, loss, train, accuracy]
    return cnn, op_list


def rnn_graph(learning_rate = 0.001, n_hidden = 128) :
    rnn = tf.Graph()
    with rnn.as_default() :
        #feed
        X = tf.placeholder(tf.float32, [None, 30, 2], name = 'X')
        Y = tf.placeholder(tf.float32, [None, 1], name = 'Y')

        #TRAINABLE variables
        cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        w = tf.get_variable(name = 'w', shape = [n_hidden, 1] ,initializer = tf.random_normal_initializer(stddev=0.01, mean=0.0))
        b = tf.get_variable(name = 'b', shape = [1] ,initializer = tf.random_normal_initializer(stddev=0.01, mean=0.0))

        #Operations

        ##<RNN>
        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        # outputs shape : (None, n_step, n_hidden)

        outputs = tf.transpose(outputs, [1, 0, 2])
        # outputs shape : (n_step, None, n_hidden)
        outputs = outputs[-1]  # 마지막 셀에서 나온 결과값만 사용합니다.
        #shape : (None, n_hidden)

        ##<basic NN>
        fc = tf.nn.sigmoid(tf.matmul(outputs, w) + b)

        #Loss function
        loss = tf.reduce_sum(tf.square(fc - Y))

        #Optimizer : Adam Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        #Accuracy
        fc = fc > 0.5
        fc = tf.to_float(fc)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(fc, Y)), name= 'accuracy')

    tf.reset_default_graph()
    op_list = [X, Y, loss, train, accuracy]

    return rnn , op_list




def runtime(name , g , op_list , datalist , ckptfile = None, total_epoches = 2, batch_size = 100) :
    ## Runtime
    import time
    X, Y, loss, train, accuracy  = op_list[0], op_list[1], op_list[2], op_list[3], op_list[4]
    train_x, train_y, valid_x, valid_y = datalist[0] ,datalist[1], datalist[2], datalist[3]

    with tf.Session(graph=g) as sess :
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        if ckptfile == None  :

            ## <training>
            total_batches = int(train_x.shape[0]/batch_size)

            print("Training Start : epoch = ", total_epoches, ", each epoch has ", total_batches," batch."  )

            for epoch in range(total_epoches) :
                print("<epoch ", epoch," >")
                total_loss = 0
                epoch_start = time.time()
                for i in range(total_batches) :
                    if len(train_x.shape) == 4:
                        batch_x  = train_x[ i*batch_size : (i+1)*batch_size ,:,:,: ]
                        batch_y  = train_y[ i*batch_size : (i+1)*batch_size ,: ]
                    elif len(train_x.shape) == 3:
                        batch_x  = train_x[ i*batch_size : (i+1)*batch_size ,:,:]
                        batch_y  = train_y[ i*batch_size : (i+1)*batch_size ,: ]

                    sess.run(train, feed_dict = {X : batch_x, Y : batch_y} )
                    l = sess.run(loss, feed_dict = {X : batch_x, Y : batch_y})
                    acc = sess.run(accuracy, feed_dict = {X : batch_x, Y : batch_y} )

                    total_loss += l

                    if i%20 == 0 or i== total_batches-1  :
                        print("batch ", i , "/",total_batches , ": loss=", l, ", accuracy=", acc)

                epoch_time = time.time() - epoch_start
                m = int(epoch_time / 60)
                s = epoch_time - m*60

                print("=>epoch ", epoch, " result: time=",m,"m ", int(s),"s , ", "total loss=", total_loss )

            print("Training Complete!")

            save_path = saver.save(sess, "./ckpt/"+name+".ckpt")
            print("Model`s Variables are saved at : ", save_path)
        else :
            saver.restore(sess, ckptfile)

        ##<validation>
        val_acc = sess.run(accuracy, feed_dict = {X : valid_x , Y : valid_y})
        print("=> Validation Set Accuracy is ", val_acc * 100 , "%.")
