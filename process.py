import os
import subprocess

import ffmpeg
import numpy as np
from tensorflow import keras

import classifier


def analyse(classifications, config):
    count = 0
    off_count = 0
    offset = 0
    results = []
    last = None

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


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def read(config):
    loglevel = 'warning' if not config['verbose'] else 'info'
    args = (
        ffmpeg
        .input(config['in_path'])
        .output('pipe:', r=1, format='rawvideo', pix_fmt='rgb24', loglevel=loglevel)
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def read_frame(process, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def run(config):
    result = []
    model = keras.models.load_model(config['model_path'])
    width, height = get_video_size(config['in_path'])
    process = read(config)

    while True:
        in_frame = read_frame(process, width, height)
        if in_frame is None:
            print('Something went wrong while reading the video!')
            break
        result.append(classifier.classify_image(in_frame, (width, height), model, config))

    process.wait()
    return result


def cut_videos(targets, config):
    out_path = config['out_path']
    if os.path.isdir(os.path.join(out_path)):
        out_dir = out_path
        basename = os.path.basename(config['in_path'])
        basename, extension = os.path.splitext(basename)
    else:
        out_dir, filename = os.path.split(out_path)
        basename, extension = os.path.splitext(filename)

    filenames = []
    loglevel = 'fatal' if not config['verbose'] else 'info'

    for index, target in enumerate(targets):
        filename = basename + '_' + str(index) + extension
        filenames.append(filename)

        (ffmpeg
            .input(
                config['in_path'],
                ss=str(target['start_time'] - config['margin_before']))
            .output(
                os.path.join(config['tmp_path'], filename),
                t=str(target['end_time'] - target['start_time'] + config['margin_after']),
                c='copy',
                loglevel=loglevel)
            .run())

    file_path = os.path.join(config['tmp_path'], 'files.txt')
    with open(file_path, 'w') as f:
        for filename in filenames:
            f.write(f'file \'{filename}\'\n')

    (ffmpeg
        .input(
            os.path.join(config['tmp_path'], 'files.txt'),
            f='concat')
        .output(
            os.path.join(out_dir, basename + extension),
            c='copy',
            loglevel=loglevel)
        .run())

    os.remove(file_path)
    for filename in filenames:
        os.remove(os.path.join(config['tmp_path'], filename))


def classify_video(config):
    print('Classifying video frames....')
    classifications = run(config)
    results = analyse(classifications, config)

    if config['verbose']:
        print('\n\n--------------------------------------------------------------------------------\n')
        for elem in results:
            print(elem)
        print('\n--------------------------------------------------------------------------------\n\n')

    print('Cutting videos....')
    cut_videos(results, config)
    print('Finished!')
