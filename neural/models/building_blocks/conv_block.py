from tensorflow.python.keras.layers import *

from models.building_blocks.conv_2d import conv_2d


def conv_block(inputs, features, kernel_size=(3, 3), pool=True):
    x = conv_2d(inputs, features, kernel_size)
    x = Dropout(0.2)(x)

    x = conv_2d(x, features, kernel_size)
    x = Dropout(0.2)(x)

    if pool:
        p = MaxPooling2D((2, 2))(x)
        return x, p
    else:
        return x
