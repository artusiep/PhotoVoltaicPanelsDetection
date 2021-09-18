from tensorflow.python.keras.layers import *


def conv_2d(inputs, features, kernel_size):
    x = Conv2D(features, kernel_size, kernel_initializer='he_normal', padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

