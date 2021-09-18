from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

from models.building_blocks.conv_block import conv_block
from models.building_blocks.conv_block_transpose import conv_block_transpose


def build_model_plus(img_width, img_height, channels_number, start_neurons=16):
    print("[LOG] Start building u-net_plus_plus model. Img_width: ", img_width, ", img_height: ", img_height,
          ", channels_number: ", channels_number, ", start_neurons: ", start_neurons)

    inputs = Input((img_width, img_height, channels_number))

    x00, p0 = conv_block(inputs, start_neurons, pool=True)

    x10, p1 = conv_block(p0, start_neurons * 2, pool=True)
    x01 = conv_block_transpose(x10, start_neurons, [x00])

    x20, p2 = conv_block(p1, start_neurons * 4, pool=True)
    x11 = conv_block_transpose(x20, start_neurons, [x10])
    x02 = conv_block_transpose(x11, start_neurons, [x00, x01])

    x30, p3 = conv_block(p2, start_neurons * 8, pool=True)
    x21 = conv_block_transpose(x30, start_neurons, [x20])
    x12 = conv_block_transpose(x21, start_neurons, [x10, x11])
    x03 = conv_block_transpose(x12, start_neurons, [x00, x01, x02])

    x40 = conv_block(p3, start_neurons * 16, pool=False)
    x31 = conv_block_transpose(x40, start_neurons * 8, [x30])
    x22 = conv_block_transpose(x31, start_neurons * 4, [x20, x21])
    x13 = conv_block_transpose(x22, start_neurons * 2, [x10, x11, x12])
    x04 = conv_block_transpose(x13, start_neurons, [x00, x01, x02, x03])

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("[LOG] U-net_plus_plus model built.")
    return model
