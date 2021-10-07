from tensorflow.python.keras.layers import *


def conv_2d_transpose(inputs, features, concatenation_list):
    x = Conv2DTranspose(features, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = concatenate(concatenation_list + [x])

    return x
