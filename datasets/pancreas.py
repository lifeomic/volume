"""
This dataset loads the KiTS Pancreas dataset into memory and serves subvolumes
"""

import argparse
import csv
import numpy as np
import os
import shutil
import sys
from collections import OrderedDict
from PIL import Image, ImageFilter

import torch
import torchvision as tv
from torch.distributions import Bernoulli, Uniform
from torch.utils.data import Dataset

from general.utils import expand_square_image, grow_image_to_square

from utils import make_xyz_gradients

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

# Patch size should be (depth, height, width)
class Pancreas(Dataset):
    def __init__(self, data_supdir, patch_size=(32,128,128), use_coordconv=True):
        self._data_supdir = data_supdir
        self._patch_size = patch_size
        self._use_coordconv = use_coordconv

        # Data and labels
        self._case_paths = None
        self._cases = None
        self._grad_x = {}
        self._grad_y = {}
        self._grad_z = {}
        self._pancreas = {}
        self._tumor = {}
        self._volumes = {}

        # Random numbers
        self._rnd_flip_x = None
        self._rnd_flip_y = None
        self._rnd_flip_z = None
        self._rnd_patch_x = None
        self._rnd_patch_y = None
        self._rnd_patch_z = None
        self._rnd_permutation = None
        self._xyz_permutations = None
        
        # Transforms
        self._transform = None
        self._label_transform = None

        self._init_data_and_labels()
        self.init_random_vals()

    def __getitem__(self, index):
        case = self._cases[index]
        self._check_load_case(case)
        self._make_transforms_now(index)
        return self._transform( self._volumes[case] ), \
                self._label_transform( self._pancreas[case] ), \
                self._case_paths[index]

    def __len__(self):
        return len(self._volumes)

    def init_random_vals(self):
        self._xyz_permutations = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1],
                [2,1,0]]
        ber = Bernoulli(0.5)
        N = self.__len__()
        self._rnd_flip_x = ber.sample((N,))
        self._rnd_flip_y = ber.sample((N,))
        self._rnd_flip_z = ber.sample((N,))
        uni = Uniform(0.0, 1.0)
        self._rnd_patch_x = uni.sample((N,))
        self._rnd_patch_y = uni.sample((N,))
        self._rnd_patch_z = uni.sample((N,))
        
    # These are stored as (depth, height, width) numpy arrays
    def _check_load_case(self, case):
        if self._volumes[case] is None:
            self._volumes[case] = np.load( pj(self._data_supdir, "imgs", case),
                    mmap_mode="r" )
            self._pancreas[case] = np.load( pj(self._data_supdir, "pancreas",
                case), mmap_mode="r" )
            self._tumor[case] = np.load( pj(self._data_supdir, "tumor", case), 
                    mmap_mode="r" )
            if self._use_coordconv:
                print("Have these already saved to disk as volumes!!")
                raise
                x,y,z = make_xyz_gradients( self._volumes[case].shape )
                self._grad_x[case] = x
                self._grad_y[case] = y
                self._grad_z[case] = z

    def _init_data_and_labels(self):
        imgs_dir = pj(self._data_supdir, "imgs")
        self._cases = sorted( [f for f in os.listdir(imgs_dir) \
                if f.endswith(".npy")] )
        self._case_paths = [pj(imgs_dir,c) for c in self._cases]
        for c in self._cases:
            self._volumes[c] = None
            self._pancreas[c] = None
            self._tumor[c] = None

    def _make_transforms_now(self, index):
        case = self._cases[index]
        volume = self._volumes[case]
        dp,ht,wd = volume.shape
        p_dp,p_ht,p_wd = self._patch_size
        x0 = int( self._rnd_patch_x[index] * (wd - p_wd) )
        y0 = int( self._rnd_patch_y[index] * (ht - p_ht) )
        z0 = int( self._rnd_patch_z[index] * (dp - p_dp) )
        x1 = x0 + p_wd
        y1 = y0 + p_ht
        z1 = z0 + p_dp

        def transform(v, p0, p1):
            x0,y0,z0 = p0
            x1,y1,z1 = p1
            v = (v[z0:z1, y0:y1, x0:x1]).astype(np.float32) / 255.0
            return torch.FloatTensor(v)
            
        self._transform = lambda x : transform(x, (x0, y0, z0), (x1, y1, z1))
        self._label_transform = lambda x : transform(x, (x0, y0, z0),
                (x1, y1, z1))


def _test_main(args):
    cfg = vars(args)
    dataset = Pancreas(cfg["data_supdir"], use_coordconv=cfg["use_coordconv"])
    N = len(dataset)
    print("Created Pancreas dataset, length %d" % N)
    output_dir = cfg["output_dir"]
    if pe(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for index in range(cfg["num_outputs"]):
        vol,label,return_path = dataset[index]
        title = os.path.splitext( os.path.basename(return_path) )[0]
        path = pj(output_dir, "%d_%s.png" % (index, title))
        vol = vol.view((-1, 1, *vol.shape[1:]))
        label = label.view((-1, 1, *label.shape[1:]))
        tv.utils.save_image(vol, path)
        label_path = pj(output_dir, "%d_%s_label.png" % (index, title))
        tv.utils.save_image(label, label_path)
        print("%d. Wrote %s, shape: %s" % (index, path, repr(vol.shape)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/Pancreas/Volumes"))
    parser.add_argument("--no-cc", "--no-coordconv", dest="use_coordconv",
            action="store_false")
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Training/volume/test_out/Pancreas"))
    parser.add_argument("-n", "--num-outputs", type=int, default=10)
    args = parser.parse_args()
    _test_main(args)

