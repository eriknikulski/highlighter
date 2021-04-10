import argparse
import json
from multiprocessing import Pool
import os
import tempfile

import process

parser = argparse.ArgumentParser(description='Highlighter for cs go clips.')
parser.add_argument('func', choices=['classify', 'build'],
                    help='The function that should be executed. '
                         'Choose either classify to extract highlights from a given video clip or '
                         'build to concatenates video clips in given folder.')
parser.add_argument('-i', '--in_path', dest='in_path', required=True,
                    help='video file path for function classify, folder path for function build '
                         'from which highlights should be selected')
parser.add_argument('-o', '--out_path', dest='out_path',
                    help='Path, with or without filename, to where highlights will be saved')
parser.add_argument('-c', '--config', dest='config', default="config.json", help='config to run')
parser.add_argument('-td', '--to-dict', dest='to_dict',
                    help='path to which highlights dict should be saved. Videos are not cut')
parser.add_argument('-dp', '--dict-path', dest='dict_path', help='path to highlights dict')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                    help='if set outputs will be verbose')


def build_config(args):
    with open(args.config) as f:
        config = json.load(f)
    config['verbose'] = args.verbose
    config['to_dict'] = args.to_dict
    config['dict_path'] = args.dict_path

    if not args.out_path:
        if 'out_path' not in config:
            raise ValueError('no output location given. Either specify in config or as argument with -o')
    else:
        config['out_path'] = args.out_path

    if not args.in_path:
        if 'in_path' not in config:
            raise ValueError('no output location given. Either specify in config or as argument with -o')
    else:
        config['in_path'] = args.in_path

    if 'tmp_path' not in config:
        config['tmp_path'] = tempfile.mkdtemp()
    return config


def classify(args):
    config = build_config(args)

    if os.path.isdir(config['in_path']):
        video_files = [f for f in os.listdir(config['in_path'])
                       if os.path.isfile(os.path.join(config['in_path'], f)) and not f.startswith('.')]
        video_files.sort()

        out_path = config['out_path']
        config['out_path'] = tempfile.mkdtemp()

        with Pool(4) as p:
            res = p.map_async(classify_single, zip(video_files, [config] * len(video_files)))
            res.get()

        config['in_path'] = config['out_path']
        config['out_path'] = out_path
        process.build_video(config)
    else:
        process.classify_video(config)


def classify_single(args):
    video, config = args
    config['in_path'] = os.path.join(config['in_path'], video)
    process.classify_video(config)


def build(args):
    config = build_config(args)
    if not os.path.isdir(config['in_path']):
        raise ValueError(f'{config.in_path} is not a valid directory')
    process.build_video(config)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.func == 'classify':
        classify(args)
    if args.func == 'build':
        build(args)
