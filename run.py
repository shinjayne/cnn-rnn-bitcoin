import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import *
from models import *

flags = tf.app.flags
FLAGS = flags.FLAGS





# when making new cnn model , you MUST select `gen_png` option
def run_cnn(name, csvpath , learning_rate, batch_size, total_epoches, ckptfile, imgdir, gen_png) :
    print("###################################")
    print("############CNN RUNNING############")
    print("###################################")


    x_data , y_data = csv_to_dataset(model = "cnn", csvpath = csvpath, fromidx = 39000)


    if gen_png == True :
        png_construct(x_data, directory = imgdir)
    cnn_x = png_to_matrix(x_data.shape[0])
    cnn_y = y_data
    print('cnn_x.shape:',cnn_x.shape, 'cnn_y.shape',cnn_y.shape)

    cnn_x , cnn_y = shuffling_data(cnn_x, cnn_y)

    ctrain_x ,ctrain_y, cvalid_x, cvalid_y = train_valid_divide(cnn_x, cnn_y)
    print('ctrain_x, ctrain_y : ',ctrain_x.shape, ctrain_y.shape, 'cvalid_x, cvalid_y : ', cvalid_x.shape, cvalid_y.shape )


    #Draw graph
    cnn_g , cnn_op= cnn_graph(learning_rate)
    #Run graph
    runtime(name = name , op_list = cnn_op, datalist = [ctrain_x ,ctrain_y, cvalid_x, cvalid_y], g = cnn_g , ckptfile =ckptfile, total_epoches = total_epoches, batch_size = batch_size )






def run_rnn(name, csvpath , learning_rate, batch_size, total_epoches, ckptfile) :
    print("###################################")
    print("############RNN RUNNING############")
    print("###################################")


    x_data , y_data = csv_to_dataset(model = "rnn", csvpath = csvpath, fromidx = 39000)
    rnn_x = x_data
    rnn_y = y_data
    print('rnn_x.shape:',rnn_x.shape, 'rnn_y.shape',rnn_y.shape)

    rtrain_x , rtrain_y, rvalid_x, rvalid_y = train_valid_divide(rnn_x, rnn_y)
    print('rtrain_x, rtrain_y : ',rtrain_x.shape, rtrain_y.shape, 'rvalid_x, rvalid_y : ', rvalid_x.shape, rvalid_y.shape )

    #Draw graph
    rnn_g , rnn_op = rnn_graph(learning_rate)
    #Run graph
    runtime(name = name ,datalist =[rtrain_x  ,rtrain_y, rvalid_x, rvalid_y], op_list = rnn_op, g = rnn_g, ckptfile=ckptfile, total_epoches = total_epoches, batch_size = batch_size)







if __name__ == "__main__":

    #Settings
    flags.DEFINE_string("name", "unnamed", "model name for ckpt file")
    flags.DEFINE_string("csvpath", "./bitcoin_ticker.csv", "model directory")
    flags.DEFINE_string("imgdir", "./x_data", "image directory")
    flags.DEFINE_boolean("gen_png", False, "generate new png files or not. extreamly lot of time consumed")

    #hyperparameters
    flags.DEFINE_integer("batch_size", 100, "batch size")
    flags.DEFINE_integer("total_epoches" , 15 , "total_epoches")
    flags.DEFINE_integer("learning_rate" , 0.001 , "learning rate")


    flags.DEFINE_string("model", "cnn", "cnn, rnn")
    #flags.DEFINE_string("mode", "train", "train, valid")

    flags.DEFINE_string("ckptfile", None, "check point file path")

    if FLAGS.model == "cnn":
        run_cnn(name = FLAGS.name, csvpath = FLAGS.csvpath , learning_rate = FLAGS.learning_rate, batch_size = FLAGS.batch_size, total_epoches = FLAGS.total_epoches, ckptfile = FLAGS.ckptfile, imgdir= FLAGS.imgdir, gen_png = FLAGS.gen_png)
    elif FLAGS.model == "rnn":
        run_rnn(name = FLAGS.name, csvpath = FLAGS.csvpath , learning_rate = FLAGS.learning_rate, batch_size = FLAGS.batch_size, total_epoches = FLAGS.total_epoches, ckptfile = FLAGS.ckptfile)
