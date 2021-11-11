import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger as logging
import json


def readlines(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_intrinsics(intrinsics_path):

    lines = readlines(intrinsics_path)
    lines = [line.split(' = ') for line in lines]
    data = {key: val for key, val in lines}

    K = np.eye(3)
    K[0, 0] = data['fx_color']
    K[1, 1] = data['fy_color']
    K[0, 2] = data['mx_color']
    K[1, 2] = data['my_color']

    # # scale intrinsics
    K[0] *= 512 / float(data['colorWidth'])
    K[1] *= 384 / float(data['colorHeight'])

    # invK = np.linalg.inv(K)

    return K


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid, scene_id):
        # intrinsics = np.loadtxt(os.path.join(filepath, scene_id, f'{scene_id}.txt'), delimiter=' ')[:3, :3]

        intrinsics = load_intrinsics(os.path.join(filepath, f'{scene_id}.txt'))

        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'sensor_data', f'frame-{vid:06d}.pose.txt'))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]
        logging.debug(f'Fragment {idx} metadata: {json.dumps(meta, indent=2, default=str)}')

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'sensor_data', f'frame-{vid:06d}.color.small.jpg')))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'sensor_data', f'frame-{vid:06d}.depth.pgm'))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid,
                                                        meta['scene'])

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        logging.debug(f"Read {len(imgs)} items from scene.")

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
