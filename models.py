import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def make_mobilenet(classes, size=(224, 224, 3)):
    input = layers.Input(shape=size)
    # why use zeropadding ?
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)),
                             name='conv1_pad')(input)
    x = layers.Conv2D(32, kernel_size=(
        3, 3),  padding='valid', strides=(2, 2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(
        2, 2), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(
        2, 2), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(
        2, 2), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = layers.DepthwiseConv2D((3, 3), strides=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(
        2, 2), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1024, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(
        2, 2), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1024, kernel_size=(1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(classes, activation='softmax')(x)

    model = keras.Model(input, out)

    return model


if __name__ == '__main__':
    model = make_mobilenet(2)
    model.summary()

    pass
