import random
import os
import subprocess

from PIL import Image
import tensorflow as tf


def rename_random(source_path, destination_path):
    for image in os.listdir(source_path):
        image_path = os.path.join(source_path, image)
        _, file_extension = os.path.splitext(image)
        os.rename(image_path, os.path.join(destination_path, str(random.getrandbits(128)) + file_extension))


def rename_images(root_path):
    for video in os.listdir(os.path.join(root_path, 'sorted')):
        if video.startswith('.'):
            continue
        video_path = os.path.join(root_path, 'sorted', video)

        rename_random(os.path.join(video_path, '/kill'), os.path.join(root_path, '/kill/'))
        rename_random(os.path.join(video_path, '/no_kill'), os.path.join(root_path, '/no_kill/'))


def extract_images(path_in, path_out, config):
    loglevel = ['-loglevel', 'fatal'] if not config['verbose'] else []
    subprocess.run([config['ffmpeg'],
                    '-i', path_in,
                    '-r', '1',
                    '-q:v', '2',
                    *loglevel,
                    os.path.join(path_out, 'frame_%04d.jpg')])


def resize_images(path):
    count = 0
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)

        with Image.open(fpath) as img:
            if img.size[1] == 1440:
                img = img.resize((1920, 1080))
                img.save(fpath)
                count += 1
    return count


def delete_corrupt_images(path):
    num_skipped = 0
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
    return num_skipped


def delete_all(path):
    count = 0
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        os.remove(fpath)
        count += 1
    return count


def cut_videos(source, targets, config):
    video_name = source[source.rfind('/') + 1:]
    extension_point = video_name.rfind('.')
    extension = video_name[extension_point:]
    video_name = video_name[:extension_point]

    filenames = []
    loglevel = ['-loglevel', 'fatal'] if not config['verbose'] else []

    for index, target in enumerate(targets):
        filename = video_name + '_' + str(index) + extension
        filenames.append(filename)
        subprocess.run([config['ffmpeg'],
                        '-ss', str(target['start_time'] - config['margin_before']),
                        '-i', source,
                        '-to', str(target['end_time'] + config['margin_after']),
                        '-codec', 'copy',
                        *loglevel,
                        os.path.join(config['out_path'], filename)])

    file_path = os.path.join(config['out_path'], 'files.txt')
    with open(file_path, 'w') as f:
        for filename in filenames:
            f.write(f'file \'{filename}\'\n')

    subprocess.run([config['ffmpeg'],
                    '-f', 'concat',
                    '-i', 'highlights/files.txt',
                    '-c', 'copy',
                    *loglevel,
                    os.path.join(config["out_path"], video_name + '.mp4')])

    os.remove(file_path)
    for filename in filenames:
        os.remove(os.path.join(config['out_path'], filename))