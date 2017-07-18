import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from random import shuffle

def csv_to_dataset(model = "cnn", csvpath = "./bitcoin_ticker.csv", fromidx = 39000) :

    def df_norm(df) :
        newdf = (df - df.mean()) /(df.max() - df.min())
        return newdf - newdf.min()

    data = pd.read_csv('./bitcoin_ticker.csv')
    data = data[data['market'] == 'korbit']
    data = data[data['rpt_key'] == 'btc_krw']
    data = data[['last', 'volume']]

    data_norm = df_norm(data)
    data_pretty = data_norm[fromidx:] # 2017년 6월 28일 데이터부터 사용

    data_pretty_values= data_pretty.values

    #plt.plot(data_pretty)
    #plt.show()

    x_data = []
    y_data = []

    if model=="cnn" :
        for i in range(30, data_pretty.shape[0]-6) :
            x_data.append( df_norm( data_pretty[i-30: i] ).values )  #30 min

            p1 = data_pretty_values[i,0]
            p2 = data_pretty_values[i+5, 0]

            result = int(p2 > p1)   # y=1 => 5분뒤 오른다  / y=0 => 5분뒤 내린다
            y_data.append(result)

        x_data = np.array(x_data, np.float32)
        y_data = np.array(y_data, np.float32)
        y_data = np.reshape(y_data, [data_pretty.shape[0]-36, 1])

    elif model=="rnn" :
        for i in range(30, data_pretty.shape[0]-6) :
            x_data.append( data_pretty[i-30: i].values )  #30 min

            p1 = data_pretty_values[i,0]
            p2 = data_pretty_values[i+5,0]

            result = int(p2 > p1)   # y=1 => 5분뒤 오른다  / y=0 => 5분뒤 내린다
            y_data.append(result)

        x_data = np.array(x_data, np.float32)
        y_data = np.array(y_data, np.float32)
        y_data = np.reshape(y_data, [data_pretty.shape[0]-36, 1])

    return x_data, y_data

## image construction
## input : numpy array with shape (batch, length, features) / directory / dpi = image size
## output : --
## doing : making png files in selected directory

def png_construct(data , directory = './x_data', dpi = 10) :

    for i in range(data.shape[0])  :
        fig = plt.figure(i , frameon=False)
        fig.set_size_inches(5,5)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.plot(data[i, :, :], linewidth=5)
        plt.ylim(0.0,1.0)

        fig.savefig(directory +'/' + str(i) + '.png', transparant = True, pad_inches = 0 , dpi = dpi)

        plt.close(fig)
        if i == 0 :
            print('<Start Constructing>')
        elif i%1000 == 0 :
            print('./x_data/' + str(i-999) + '.png to ' +'./x_data/' + str(i) + ".png have been saved")
        elif i == data.shape[0]-1 :
            print('<Complete Image Constructing> : Total ',data.shape[0],' images into ', directory)

#image to matrix

#input : number of images, directory
#output : numpy array of images

def png_to_matrix(num ,directory = './x_data'):

    x_data = []

    for i in range(num) :
        img = mpimg.imread(directory+'/'+ str(i) +'.png') # type : np.float32

        x_data.append(img)

        if i == 0 :
            print("<Start Transfer> : shape = ", img.shape)

        elif i == num-1 :
            print('<Complete image to Matrix transfer>  : Total ',num ,' matrix into list => shape = ', img.shape)

    x_data = np.array(x_data, dtype=np.float32)
    return x_data


class DiffLengthError(Exception):
    pass
    #print('batch length of x_data and y_data are not same')



# Shuffling datas
#input : x_data, y_data with same length
def shuffling_data(x_data, y_data) :
    if x_data.shape[0] != y_data.shape[0]:
        raise DiffLengthError
        return 0
    else :
        x_data = list(x_data)
        y_data = list(y_data)
        # Given list1 and list2
        list1_shuf = []
        list2_shuf = []
        index_shuf = list(range(len(x_data)))
        shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(x_data[i])
            list2_shuf.append(y_data[i])

        x_data = np.array(list1_shuf)
        y_data = np.array(list2_shuf)

        return x_data, y_data


def train_valid_divide(x_data, y_data, per = 0.8) :
    if x_data.shape[0] != y_data.shape[0]:
        raise DiffLengthError
        return 0
    else :
        if len(x_data.shape) ==4 :
            l = x_data.shape[0]
            tl = int(l * per )
            train_x = x_data[:tl, :,:,:]
            train_y = y_data[:tl, :]

            valid_x = x_data[tl: ,:,:,:]
            valid_y = y_data[tl:, :]

            return train_x, train_y, valid_x, valid_y
        elif len(x_data.shape) ==3 :
            l = x_data.shape[0]
            tl = int(l * per )
            train_x = x_data[:tl, :,:]
            train_y = y_data[:tl, :]

            valid_x = x_data[tl: ,:,:]
            valid_y = y_data[tl:, :]

            return train_x, train_y, valid_x, valid_y
