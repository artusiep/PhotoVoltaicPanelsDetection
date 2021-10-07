from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


def build_model(img_width, img_height, channels_number, start_neurons=16):
    inputs = Input((img_width, img_height, channels_number))

    x00 = BatchNormalization()(inputs)
    x00 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    x00 = Dropout(0.2)(x00)
    x00 = BatchNormalization()(x00)
    x00 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D((2, 2))(x00)

    x10 = BatchNormalization()(p0)
    x10 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    x10 = Dropout(0.2)(x10)
    x10 = BatchNormalization()(x10)
    x10 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D((2, 2))(x10)

    x20 = BatchNormalization()(p1)
    x20 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = Dropout(0.2)(x20)
    x20 = BatchNormalization()(x20)
    x20 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D((2, 2))(x20)

    x30 = BatchNormalization()(p2)
    x30 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = Dropout(0.2)(x30)
    x30 = BatchNormalization()(x30)
    x30 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D((2, 2))(x30)

    x40 = BatchNormalization()(p3)
    x40 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = Dropout(0.2)(x40)
    x40 = BatchNormalization()(x40)
    x40 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = Dropout(0.2)(x40)
    p4 = MaxPooling2D((2, 2))(x40)

    x50 = BatchNormalization()(p4)
    x50 = Conv2D(start_neurons * 32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x50)
    x50 = Dropout(0.2)(x50)
    x50 = BatchNormalization()(x50)
    x50 = Conv2D(start_neurons * 32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x50)
    x50 = Dropout(0.2)(x50)
    p6 = MaxPooling2D((2, 2))(x50)

    x60 = BatchNormalization()(p6)
    x60 = Conv2D(start_neurons * 64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x60)
    x60 = Dropout(0.2)(x60)
    x60 = BatchNormalization()(x60)
    x60 = Conv2D(start_neurons * 64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x60)
    x60 = Dropout(0.2)(x60)

    # -------

    x51 = Conv2DTranspose(start_neurons * 32, kernel_size=(2, 2), strides=(2, 2), padding='same')(x60)
    x51 = concatenate([x50, x51])
    x51 = BatchNormalization()(x51)
    x51 = Conv2D(start_neurons * 32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x51)
    x51 = BatchNormalization()(x51)
    x51 = Conv2D(start_neurons * 32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x51)
    x51 = Dropout(0.2)(x51)

    x41 = Conv2DTranspose(start_neurons * 16, kernel_size=(2, 2), strides=(2, 2), padding='same')(x51)
    x41 = concatenate([x40, x41])
    x41 = BatchNormalization()(x41)
    x41 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x41)
    x41 = BatchNormalization()(x41)
    x41 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x41)
    x41 = Dropout(0.2)(x41)

    x31 = Conv2DTranspose(start_neurons * 8, kernel_size=(2, 2), strides=(2, 2), padding='same')(x41)
    x31 = concatenate([x30, x31])
    x31 = BatchNormalization()(x31)
    x31 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = BatchNormalization()(x31)
    x31 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = Dropout(0.2)(x31)

    x22 = Conv2DTranspose(start_neurons * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(x31)
    x22 = concatenate([x20, x22])
    x22 = BatchNormalization()(x22)
    x22 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = Dropout(0.2)(x22)

    x13 = Conv2DTranspose(start_neurons * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(x22)
    x13 = concatenate([x10, x13])
    x13 = BatchNormalization()(x13)
    x13 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = BatchNormalization()(x13)
    x13 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = Dropout(0.2)(x13)

    x04 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x13)
    x04 = concatenate([x00, x04])
    x04 = BatchNormalization()(x04)
    x04 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = BatchNormalization()(x04)
    x04 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = Dropout(0.2)(x04)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(x04)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
