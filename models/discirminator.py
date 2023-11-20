import numpy as np

from tensorflow import keras

IMG_SHAPE = (128, 128, 1)
NUM_CLASSES = 2


def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    use_bias=False,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = keras.layers.Dropout(drop_value)(x)
    return x


def build_d() -> keras.Model:
    # first imput -> images
    imgs_input = keras.Input(shape=IMG_SHAPE, name="images_input")
    # second imput -> labels
    label_input = keras.Input(shape=(1,), dtype="int32", name="label_input")
    label_embedding = keras.layers.Embedding(
        NUM_CLASSES, np.prod(IMG_SHAPE), input_length=1
    )(label_input)
    label_embedding = keras.layers.Flatten()(label_embedding)
    label_embedding = keras.layers.Reshape(target_shape=IMG_SHAPE)(label_embedding)

    x = keras.layers.Concatenate(axis=-1)([imgs_input, label_embedding])
    x = conv_block(
        x,
        filters=64,
        activation=keras.layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(1, 1),
        use_bias=True,
    )
    x = conv_block(
        x,
        filters=128,
        activation=keras.layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        use_bias=False,
        use_bn=True,
    )
    x = conv_block(
        x,
        filters=256,
        activation=keras.layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        use_bias=False,
        use_bn=True,
    )
    x = conv_block(
        x,
        filters=512,
        activation=keras.layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        use_bias=False,
        use_bn=True,
    )
    x = conv_block(
        x,
        filters=1024,
        activation=keras.layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        use_bias=False,
        use_bn=True,
    )
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    output = keras.layers.Dense(units=1, name="output_logit")(x)
    return keras.Model(
        inputs=[imgs_input, label_input], outputs=output, name="DISCRIMINATOR"
    )
