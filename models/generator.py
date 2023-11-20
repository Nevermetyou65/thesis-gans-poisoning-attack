import numpy as np

from tensorflow import keras


Z_DIM = 128
INITIAL_SHAPE = (8, 8, 1024)
NUM_CLASSES = 2
EMBED_LABEL_SHAPE = (8, 8, 1)


def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_upsize=False,
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    if use_upsize:
        x = keras.layers.UpSampling2D(up_size)(x)
    x = keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = keras.layers.Dropout(drop_value)(x)
    return x


def build_g() -> keras.Model:
    # first input -> latent vector from normal distribution
    noise_input = keras.Input(shape=Z_DIM, name="noise_input")  # input 1
    noise = keras.layers.Dense(
        units=np.prod(INITIAL_SHAPE), use_bias=False, name="fully_connected"
    )(noise_input)
    noise = keras.layers.BatchNormalization()(noise)
    noise = keras.layers.ReLU()(noise)
    noise = keras.layers.Reshape(target_shape=INITIAL_SHAPE)(noise)
    # second input -> labels
    label_input = keras.Input(shape=(1,), dtype="int32", name="label_input")  # input 2
    label_embedding = keras.layers.Embedding(
        NUM_CLASSES, np.prod(EMBED_LABEL_SHAPE), input_length=1
    )(label_input)
    label_embedding = keras.layers.Flatten()(label_embedding)
    label_embedding = keras.layers.Reshape(target_shape=EMBED_LABEL_SHAPE)(
        label_embedding
    )

    x = keras.layers.Concatenate(axis=-1)([noise, label_embedding])
    x = upsample_block(
        x,
        filters=512,
        activation=keras.layers.ReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_upsize=True,
        use_bias=False,
        use_bn=True,
    )
    x = upsample_block(
        x,
        filters=256,
        activation=keras.layers.ReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_upsize=True,
        use_bias=False,
        use_bn=True,
    )
    x = upsample_block(
        x,
        filters=128,
        activation=keras.layers.ReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_upsize=True,
        use_bias=False,
        use_bn=True,
    )
    x = upsample_block(
        x,
        filters=64,
        activation=keras.layers.ReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_upsize=True,
        use_bias=False,
        use_bn=True,
    )
    output = upsample_block(
        x,
        filters=1,
        activation=keras.layers.Activation("tanh"),
        kernel_size=(5, 5),
        strides=(1, 1),
        use_upsize=False,
        use_bias=True,
        use_bn=False,
    )
    return keras.Model(
        inputs=[noise_input, label_input], outputs=output, name="GENERATOR"
    )
