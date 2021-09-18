from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


# from models.building_blocks.conv_2d_transpose import conv_2d_transpose
# from models.building_blocks.conv_block import conv_2d, conv_block
#
#
# def build_model(img_width, img_height, channels_number, start_neurons=16):
#     print("[LOG] Start building dense_u-net model. Img_width: ", img_width, ", img_height: ", img_height,
#           ", channels_number: ", channels_number, ", start_neurons: ", start_neurons)
#
#     inputs = Input((img_width, img_height, channels_number))
#
#     x00 = dense_block(start_neurons * 1, inputs)
#     pool0, x00 = transition_block(start_neurons * 1, x00)
#
#     x10 = dense_block(start_neurons * 2, pool0)
#     pool1, x10 = transition_block(start_neurons * 2, x10)
#
#     x20 = dense_block(start_neurons * 4, pool1)
#     pool2, x20 = transition_block(start_neurons * 4, x20)
#
#     x30 = dense_block(start_neurons * 8, pool2)
#     pool3, x30 = transition_block(start_neurons * 8, x30)
#
#     x40 = dense_block(start_neurons * 16, pool3)
#
#     x31 = conv_2d_transpose(x40, start_neurons * 8, [x30])
#     x31 = dense_block(start_neurons * 8, x31)
#
#     x22 = conv_2d_transpose(x31, start_neurons * 4, [x20])
#     x22 = dense_block(start_neurons * 4, x22)
#
#     x13 = conv_2d_transpose(x22, start_neurons * 2, [x10])
#     x13 = dense_block(start_neurons * 2, x13)
#
#     x04 = conv_2d_transpose(x13, start_neurons, [x00])
#     x04 = dense_block(start_neurons * 1, x04)
#
#     outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     print("[LOG] Dense_u-net model built.")
#     return model
#
#
# def dense_block(features, inputs_0):
#     dense_conv_layer_0 = conv_block(inputs_0, features, (1, 1), False)
#
#     inputs_1 = concatenate([inputs_0, dense_conv_layer_0])
#     dense_conv_layer_1 = conv_block(inputs_1, features, (1, 1), False)
#
#     inputs_2 = concatenate([inputs_0, dense_conv_layer_0, dense_conv_layer_1])
#     dense_conv_layer_2 = conv_block(inputs_2, features, (1, 1), False)
#
#     inputs_3 = concatenate([inputs_0, dense_conv_layer_0, dense_conv_layer_1, dense_conv_layer_2])
#     dense_conv_layer_3 = conv_block(inputs_3, features, (1, 1), False)
#
#     return dense_conv_layer_3
#
#
# def transition_block(features, inputs):
#     x = conv_2d(inputs, features, (1, 1))
#     p = MaxPooling2D((2, 2))(x)
#
#     return p, x

# ------------------------------------------------------------
# | Aleks original code. Validate if current one works fine. |
# ------------------------------------------------------------

def build_model(img_width, img_height, channels_number, start_neurons=16):
    print("[LOG] Start building dense_u-net model. Img_width: ", img_width, ", img_height: ", img_height,
          ", channels_number: ", channels_number, ", start_neurons: ", start_neurons)

    inputs = Input((img_width, img_height, channels_number))

    x00 = DenseBlock(start_neurons * 1, inputs)
    pool0, x00 = TransitionBlock(start_neurons * 1, x00)

    x10 = DenseBlock(start_neurons * 2, pool0)
    pool1, x10 = TransitionBlock(start_neurons * 2, x10)

    x20 = DenseBlock(start_neurons * 4, pool1)
    pool2, x20 = TransitionBlock(start_neurons * 4, x20)

    x30 = DenseBlock(start_neurons * 8, pool2)
    pool3, x30 = TransitionBlock(start_neurons * 8, x30)

    x40 = DenseBlock(start_neurons * 16, pool3)

    x31 = Conv2DTranspose(start_neurons * 8, kernel_size=(2, 2), strides=(2, 2), padding='same')(x40)
    x31 = concatenate([x30, x31])
    x31 = DenseBlock(start_neurons * 8, x31)

    x22 = Conv2DTranspose(start_neurons * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(x31)
    x22 = concatenate([x20, x22])
    x22 = DenseBlock(start_neurons * 4, x22)

    x13 = Conv2DTranspose(start_neurons * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(x22)
    x13 = concatenate([x10, x13])
    x13 = DenseBlock(start_neurons * 2, x13)

    x04 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x13)
    x04 = concatenate([x00, x04])
    x04 = DenseBlock(start_neurons * 1, x04)

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("[LOG] Dense_u-net model built.")
    return model


def DenseBlock(features, inputs_0):
    dense_conv_layer_0 = DenseConvLayer(features, inputs_0)

    inputs_1 = concatenate([inputs_0, dense_conv_layer_0])
    dense_conv_layer_1 = DenseConvLayer(features, inputs_1)

    inputs_2 = concatenate([inputs_0, dense_conv_layer_0, dense_conv_layer_1])
    dense_conv_layer_2 = DenseConvLayer(features, inputs_2)

    inputs_3 = concatenate([inputs_0, dense_conv_layer_0, dense_conv_layer_1, dense_conv_layer_2])
    dense_conv_layer_3 = DenseConvLayer(features, inputs_3)

    return dense_conv_layer_3


def DenseConvLayer(features, inputs):
    batch_normalization_0 = BatchNormalization()(inputs)
    conv_0 = Conv2D(features, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(
        batch_normalization_0)

    batch_normalization_1 = BatchNormalization()(conv_0)
    conv_1 = Conv2D(features, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        batch_normalization_1)

    dropout = Dropout(0.2)(conv_1)

    return dropout


def TransitionBlock(features, inputs):
    batch_normalization_0 = BatchNormalization()(inputs)
    conv_0 = Conv2D(features, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(
        batch_normalization_0)

    max_pooling = MaxPooling2D((2, 2))(conv_0)

    return max_pooling, conv_0
