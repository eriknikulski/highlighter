import os
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import image_extractor


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def classify(config):
    image_size = (480, 270)
    results = []

    model = keras.models.load_model(config['model_path'])

    image_list = [image for image in os.listdir(config['tmp_path'])]
    image_list.sort(key=natural_keys)

    for image in image_list:
        image_path = str(os.path.join(config['tmp_path'], image))
        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]

        if config['verbose']:
            if score > 0.5:
                decision = "NO KILL"
            else:
                decision = "KILL"

            print(
                "%s     This %s is %.2f percent kill and %.2f percent no kill."
                % (decision, image, 100 * (1 - score), 100 * score)
            )

        results.append(1 - score)
    return results


def analyse(classifications, config):
    count = 0
    off_count = 0
    results = []
    last = None
    offset = 0

    if len(classifications) > config['min_slice']:
        offset = int(config['slicing_percentage'] * len(classifications))
        classifications = classifications[offset:]

    for index, score in enumerate(classifications):
        index += offset
        if score > 0.5:
            count += 1
        else:
            if count >= config['min_single_kill_trigger']:
                if off_count < config['off_kill_limit'] and last:
                    last['end_time'] = index
                else:
                    if last:
                        results.append(last)
                    last = {
                        'type': 'one' if count < config['min_multi_kill_trigger'] else 'multiple',
                        'start_time': index - count,
                        'end_time': index}
                count = 0
                off_count = 0
            off_count += 1
    results.append(last)
    return results


def classify_video(video_path, config):
    tmp_path = config['tmp_path']
    verbose = config['verbose']
    print('Start extraction.... of {} to {}'.format(video_path, tmp_path))
    image_extractor.extract_images(video_path, tmp_path, config)
    if verbose:
        print('Finished extraction')
        print('Start corruption check....')
    deleted = image_extractor.delete_corrupt_images(tmp_path)
    if verbose:
        print('Deleted {} images'.format(deleted))
        print('Check for correct image format....')
    resized = image_extractor.resize_images(tmp_path)
    if verbose:
        print('Resized {} images'.format(resized))
    print('Start classification....')
    classifications = classify(config)
    results = analyse(classifications, config)

    if verbose:
        print('\n\n--------------------------------------------------------------------------------\n')
        for elem in results:
            print(elem)
        print('\n--------------------------------------------------------------------------------\n\n')

    if config['cleanup']:
        deletions = image_extractor.delete_all(tmp_path)
        if verbose:
            print('Deleted {} images in tmp'.format(deletions))

    print('Cutting videos....')
    image_extractor.cut_videos(video_path, results, config)
    print('Finished!')


def train(image_source_path):
    epochs = 30
    # epochs = 10
    # scaling = 4
    # image_size = (int(1920 / scaling), int(1080 / scaling))
    image_size = (480, 270)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_source_path,
        image_size=image_size,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_source_path,
        image_size=image_size,
        validation_split=0.2,
        subset="validation",
        seed=1337,
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model(input_shape=image_size + (3,))
    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint("models/saves/save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
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

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)