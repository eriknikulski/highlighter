import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def classify_image(img_array, old_image_size, model, config):
    image_size = (int(old_image_size[0] / 4), int(old_image_size[1] / 4))

    bin_size = old_image_size[0] // image_size[0]
    img_array = img_array.reshape((image_size[1], bin_size, image_size[0], bin_size, 3)).max(3).max(1)

    img_array = tf.convert_to_tensor(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0][0]

    if config['verbose']:
        decision = 'KILL' if score < 0.5 else 'NO KILL'
        print('{}     This image is {:.2%} percent kill and {:.2%} percent no kill.'.format(
                decision, 1 - score, score))
    return 1 - score


def train(image_source_path):
    epochs = 30
    image_size = (270, 480)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_source_path,
        image_size=image_size,
        validation_split=0.2,
        subset='training',
        seed=1337,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_source_path,
        image_size=image_size,
        validation_split=0.2,
        subset='validation',
        seed=1337,
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model(input_shape=image_size + (3,))
    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint('models/saves/save_at_{epoch}.h5'),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )


def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    left_crop = int(input_shape[0] * 0.5)
    bottom_crop = int(input_shape[1] * 0.5)

    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.Cropping2D(cropping=((0, bottom_crop), (left_crop, 0)))(x)

    x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    train('images')
