import sys
sys.path.append('.')

import time
import os
from tools.tsdf_fusion.fusion import *
import pickle
import argparse
from tqdm import tqdm
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--tsdf_folder_path", help="path to TSDF files")
    parser.add_argument("--frames_file_path", help="Path to file with frames to use")
    parser.add_argument('--test', action='store_true',
                        help='prepare the test set')
    return parser.parse_args()


args = parse_args()

def save_fragment_pkl(args, scene, ids):
    fragments = []
    print('segment: process scene {}'.format(scene))

    with open(os.path.join(args.tsdf_folder_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    if not os.path.exists(os.path.join(args.tsdf_folder_path, scene, 'fragments', str(0))):
        os.makedirs(os.path.join(args.tsdf_folder_path, scene, 'fragments', str(0)))
    fragments.append({
        'scene': scene,
        'fragment_id': 0,
        'image_ids': ids,
        'vol_origin': tsdf_info['vol_origin'],
        'voxel_size': tsdf_info['voxel_size'],
    })

    with open(os.path.join(args.tsdf_folder_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)


def readlines(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def process(args):
    frames = {}
    for line in readlines(args.frames_file_path):
        data = line.strip().split()
        frames[data[0]] = [int(i) for i in data[1:]]
    for scene in tqdm(list(frames.keys())):
        print('read from disk')
        save_fragment_pkl(args, scene, frames[scene])


def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.tsdf_folder_path))
    if not args.test:
        splits = ['train', 'val']
    else:
        splits = ['test']
    for split in splits:
        fragments = []
        with open(os.path.join(args.tsdf_folder_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.tsdf_folder_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.tsdf_folder_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


if __name__ == "__main__":
    process(args)

    if args.dataset == 'scannet':
        generate_pkl(args)
