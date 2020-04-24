import tensorflow as tf
import tensorflow_datasets as tfd

from adversarial.VAEGAN import VAEGAN


def make_encoder(input_shape, code_size: int):
    input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(input_layer)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units=code_size * 2)(x)
    return tf.keras.Model(inputs=input_layer, outputs=outputs)


def make_decoder(z_shape):
    input_layer = tf.keras.layers.Input(z_shape)
    x = tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu")(input_layer)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu")(x)
    output_layer = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME",
                                                   activation="sigmoid")(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def make_discriminator(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(input_layer)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    intermediate_output = x
    output_layer = tf.keras.layers.Dense(units=1, activation=None)(x)
    return tf.keras.Model(inputs=input_layer, outputs=[output_layer, intermediate_output])


def extract_and_convert_image(inputs):
    image, label = inputs["image"], inputs["label"]
    image = tf.cast(image, tf.float32) / 255.0
    return image


def main():
    input_shape = (28, 28, 1)
    code_size = 128
    batch_size = 64

    encoder = make_encoder(input_shape, code_size=code_size)
    decoder = make_decoder([code_size])
    discriminator = make_discriminator(input_shape)

    vaegan = VAEGAN(encoder=encoder,
                    decoder=decoder,
                    discriminator=discriminator)

    dataset, info = tfd.load("fashion_mnist", with_info=True)
    train_count = info.splits["train"].num_examples
    test_count = info.splits["test"].num_examples
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    train_dataset = train_dataset.shuffle(train_count)
    train_dataset = train_dataset.map(extract_and_convert_image)
    train_dataset = train_dataset.batch(batch_size).prefetch(-1)

    test_dataset = test_dataset.shuffle(test_count)
    test_dataset = test_dataset.map(extract_and_convert_image)
    test_dataset = test_dataset.batch(batch_size).prefetch(-1)

    train_steps = (train_count // batch_size) + 1
    validation_steps = (test_count // batch_size) + 1

    x = next(iter(test_dataset))
    y = vaegan(x)

    x = x[0].numpy()
    y = y[0].numpy()

    import cv2
    cv2.imshow("x", cv2.resize(x, (256, 256)))
    cv2.imshow("y", cv2.resize(y, (256, 256)))
    cv2.waitKey(1)

    vaegan.fit(x=train_dataset,
               epochs=100,
               steps_per_epoch=train_steps,
               validation_data=test_dataset,
               validation_steps=validation_steps)

    x = next(iter(test_dataset))
    y = vaegan(x)
    x = x[0].numpy()
    y = y[0].numpy()
    cv2.imshow("x", cv2.resize(x, (256, 256)))
    cv2.imshow("y", cv2.resize(y, (256, 256)))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
