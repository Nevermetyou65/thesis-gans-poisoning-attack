from tensorflow import keras
from tensorflow.keras.applications import (
    vgg16,
    resnet_v2,
    convnext,
    mobilenet_v2,
    inception_v3,
)
from tensorflow.keras import layers


def build_model(model_name, image_shape):
    if model_name == "vgg16":
        base_model = vgg16.VGG16(include_top=False, input_shape=image_shape)
    if model_name == "resnet50v2":
        base_model = resnet_v2.ResNet50V2(include_top=False, input_shape=image_shape)
    if model_name == "mobilenetv2_1.00_128":
        base_model = mobilenet_v2.MobileNetV2(
            include_top=False, input_shape=image_shape
        )
    if model_name == "inception_v3":
        base_model = inception_v3.InceptionV3(
            include_top=False, input_shape=image_shape
        )
    if model_name == "convnext_tiny":
        base_model = convnext.ConvNeXtTiny(include_top=False, input_shape=image_shape)

    base_model.trainable = False
    bool_ = model_name == "vgg16"
    transition = layers.Flatten() if bool_ else layers.GlobalAveragePooling2D()
    classifier = keras.Sequential(
        [
            transition,
            layers.Dense(512, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(128, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(64, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="classifier",
    )

    x_input = keras.Input(image_shape, name="input")
    if model_name == "vgg16":
        x = vgg16.preprocess_input(x_input)
        x = layers.Rescaling(scale=1.0 / 255)(x)
    if model_name == "resnet50v2":
        x = resnet_v2.preprocess_input(x_input)
    if model_name == "mobilenetv2_1.00_128":
        x = mobilenet_v2.preprocess_input(x_input)
    if model_name == "inception_v3":
        x = inception_v3.preprocess_input(x_input)
    if model_name == "convnext_tiny":
        x = x_input
    x = base_model(x, training=False)
    output = classifier(x)

    return keras.Model(input, output, name="using-pretrained-" + model_name)
