from tensorflow.python.keras.layers import *

from models.building_blocks.conv_2d import conv_2d
from models.building_blocks.conv_2d_transpose import conv_2d_transpose


def conv_block_transpose(inputs, features, concatenation_list):
    x = conv_2d_transpose(inputs, features, concatenation_list)

    x = conv_2d(x, features, (3, 3))
    x = conv_2d(x, features, (3, 3))

    x = Dropout(0.2)(x)

    return x

