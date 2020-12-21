import argparse
import json
import os

import process

parser = argparse.ArgumentParser(description='Highlighter for cs go clips.')
parser.add_argument('func', choices=['classify', 'build'],
                    help='The function that should be executed. '
                         'Choose either classify to extracts highlights from a given video clip or '
                         'build to concatenates video clips in given folder.')
parser.add_argument('-i', '--in_path', dest='in_path', required=True,
                    help='video file path for function classify, folder path for function build '
                         'from which highlights should be selected')
parser.add_argument('-o', '--out_path', dest='out_path',
                    help='Path, with or without filename, to where highlights will be saved')
parser.add_argument('-c', '--config', dest='config', required=True, help='config to run')
parser.add_argument('-td', '--to-dict', dest='to_dict',
                    help='path to which highlights dict should be saved. Videos are not cut')
parser.add_argument('-dp', '--dict-path', dest='dict_path', help='path to highlights dict')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                    help='if set outputs will be verbose')


def classify(args):
    with open(args.config) as f:
        config = json.load(f)
    config['verbose'] = args.verbose
    config['to_dict'] = args.to_dict
    config['dict_path'] = args.dict_path

    if not args.out_path:
        if 'out_path' not in config:
            raise ValueError('no output location given. Ether specify in config or as argument with -o')
    else:
        config['out_path'] = args.out_path

    if not args.in_path:
        if 'in_path' not in config:
            raise ValueError('no output location given. Ether specify in config or as argument with -o')
    else:
        config['in_path'] = args.in_path

    basename = os.path.basename(config['in_path'])
    if basename == '':
        raise ValueError('input location needs to include filename')

    process.classify_video(config)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.func == 'classify':
        classify(args)
    if args.func == 'build':
        pass
