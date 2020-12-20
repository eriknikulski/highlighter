import argparse
import json

import classifier

parser = argparse.ArgumentParser(description='Highlighter for cs go clips.')
parser.add_argument('-c', '--config', dest='config', required=True, help='config to run')
parser.add_argument('-f', '--file', dest='file', required=True,
                    help='video file from which highlights should be selected')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                    help='if set outputs will be verbose')


def main(video_path, config_path, verbose):
    with open(config_path) as f:
        config = json.load(f)
    config['verbose'] = verbose
    classifier.classify_video(video_path, config)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.file, args.config, args.verbose)