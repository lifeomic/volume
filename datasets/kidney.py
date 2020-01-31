"""
This dataset loads the KiTS Kidney dataset into memory and serves subvolumes
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
import torch.nn.functional as F
from torch.distributions import Bernoulli, Uniform
from torch.utils.data import DataLoader, Dataset

from general.utils import expand_square_image, grow_image_to_square

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volume_utils import make_xyz_gradients

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_Kidney_loaders(cfg, loaders="train_and_test"):
    ht,wd,dp = cfg["patch_depth"], cfg["patch_height"], cfg["patch_width"]
    ds_kwargs = { "data_supdir" : cfg["kidney_data_supdir"],
            "output_cat" : cfg["kidney_output_cat"],
            "patch_size" : (ht,wd,dp) }
    for k in ["use_coordconv", "train_valid_split"]:
        ds_kwargs[k] = cfg[k]
    dl_kwargs = { "num_workers" : cfg["num_workers"],
            "batch_size" : cfg["batch_size"] }

    test_dataset = Kidney(mode="valid", return_path=True, **ds_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)
    if loaders=="train_and_test":
        train_dataset = Kidney(mode="train", **ds_kwargs)
        train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
        return train_loader,test_loader

    if loaders!="test":
        raise RuntimeError("Unrecognized value for loaders, %s" % loaders)
    return test_loader


# Patch size should be (depth, height, width)
class Kidney(Dataset):
    def __init__(self, data_supdir, mode="train", patch_size=(32,128,128),
            output_cat="kidney", use_coordconv=True, train_valid_split=0.85,
            return_path=False):
        self._data_supdir = data_supdir
        self._mode = mode
        self._patch_size = patch_size
        self._return_path = return_path
        self._train_valid_split = train_valid_split
        self._use_coordconv = use_coordconv

        # Data and labels
        self._case_paths = None
        self._cases = None
        self._grad_x = {}
        self._grad_y = {}
        self._grad_z = {}
        self._kidney = {}
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
        self._simple_transform = None
        self._simple_label_transform = None

        self._init_data_and_labels()
        self._make_transforms()
        self.init_random_vals()

    def __getitem__(self, index):
        case = self._cases[index]
        self._check_load_case(case)
        self._make_transforms_now(index)
        data = self._transform( self._volumes[case] )
        if self._use_coordconv:
            grad_x = self._transform( self._grad_x[case] )
            grad_y = self._transform( self._grad_y[case] )
            grad_z = self._transform( self._grad_z[case] )
            data = torch.cat([data, grad_x, grad_y, grad_z], dim=0)
        if self._return_path:
            return data, \
                    self._label_transform( self._kidney[case] ), \
                    self._case_paths[index]
        return data, self._label_transform( self._kidney[case] )

    def __len__(self):
        return len(self._volumes)

    def get_class_weights(self, min_wt=1.0): # TODO
        return 5.0,1.0

    def get_name(self):
        return "Kidney"

    # p0, p1 should be given as (depth, height, width), in voxel coordinates
    def get_patch(self, index, p0, p1):
        case = self._cases[index]
        self._check_load_case(case)
        patch = self._simple_transform( self._volumes[case], p0, p1 )
        if self._use_coordconv:
            grad_x = self._simple_transform( self._grad_x[case], p0, p1 )
            grad_y = self._simple_transform( self._grad_y[case], p0, p1 )
            grad_z = self._simple_transform( self._grad_z[case], p0, p1 )
            patch = torch.cat([patch, grad_x, grad_y, grad_z], dim=0)
        return patch

    def get_path(self, index):
        return self._case_paths[index]

    def get_shape(self, index):
        case = self._cases[index]
        self._check_load_case(case)
        return self._volumes[case].shape

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
            for dic,dtype in zip([self._volumes, self._kidney, self._tumor],
                    ["imgs", "kidney", "tumor"]):
                dic[case] = np.load( pj(self._data_supdir, dtype, case),
                    mmap_mode="r" )
            if self._use_coordconv:
                for dic,dtype in zip([self._grad_x, self._grad_y, self._grad_z],
                        ["grad_x", "grad_y", "grad_z"]):
                    dic[case] = np.load( pj(self._data_supdir, dtype, case),
                        mmap_mode="r" )

    def _init_data_and_labels(self):
        imgs_dir = pj(self._data_supdir, "imgs")
        all_cases = sorted( [f for f in os.listdir(imgs_dir) \
                if f.endswith(".npy")] )
        all_case_paths = [pj(imgs_dir,c) for c in all_cases]
        self._set_train_and_valid(all_case_paths)

    # p0, p1 have coordinates in order (x,y,z)
    def _make_transforms(self):
        def transform(v, p0, p1):
            p_wd,p_ht,p_dp = [p[1] - p[0] for p in zip(p0, p1)]
            (x0,y0,z0),(x1,y1,z1) = p0, p1
            v = (v[z0:z1, y0:y1, x0:x1]).astype(np.float32) / 255.0
            x = torch.FloatTensor(v.copy()).view((1, 1, *v.shape))
            if v.shape != (p_dp,p_ht,p_wd):
                x = F.interpolate(x, (p_dp,p_ht,p_wd), mode="trilinear",
                        align_corners=False)
            return x.view((-1, *x.shape[2:]))
        # Without copy(), get "Trying to resize storage that is not resizable"
        # error
            
        def label_transform(v, p0, p1):
            p_wd,p_ht,p_dp = [p[1] - p[0] for p in zip(p0, p1)]
            (x0,y0,z0),(x1,y1,z1) = p0, p1
            v = (v[z0:z1, y0:y1, x0:x1]).astype(np.float32) / 255.0
            x = torch.FloatTensor(v.copy()).view((1, 1, *v.shape))
            if v.shape != (p_dp,p_ht,p_wd):
                x = F.interpolate(x, (p_dp,p_ht,p_wd), mode="nearest")
            return x.view((-1, *x.shape[2:]))

        self._simple_transform = transform
        self._simple_label_transform = label_transform

    # !!! TODO !!! gradients need to be registered with TRUE WORLD COORDINATES
    def _make_transforms_now(self, index):
        case = self._cases[index]
        volume = self._volumes[case]
        dp,ht,wd = volume.shape
        p_dp,p_ht,p_wd = self._patch_size
        if self._mode=="train":
            x0 = max(0, int( self._rnd_patch_x[index] * (wd - p_wd) ) )
            y0 = max(0, int( self._rnd_patch_y[index] * (ht - p_ht) ) )
            z0 = max(0, int( self._rnd_patch_z[index] * (dp - p_dp) ) )
        else:
            x0 = (wd - p_wd) // 2
            y0 = (ht - p_ht) // 2
            z0 = (dp - p_dp) // 2
        x1,y1,z1 = x0 + p_wd, y0 + p_ht, z0 + p_dp

        self._transform = lambda x : self._simple_transform(x, (x0, y0, z0),
                (x1, y1, z1))
        self._label_transform = lambda x : self._simple_label_transform(x,
                (x0, y0, z0), (x1, y1, z1))

    def _set_train_and_valid(self, all_case_paths):
        self._cases = []
        self._case_paths = []
        if self._train_valid_split < 1.0 and self._mode != "test":
            train_ct = 0
            for i in range( len(all_case_paths) ):
                if train_ct/(i+1) < self._train_valid_split:
                    if self._mode == "train":
                        self._case_paths.append( all_case_paths[i] )
                    train_ct += 1
                elif self._mode == "valid":
                    self._case_paths.append( all_case_paths[i] )
        else:
            self._case_paths = all_case_paths

        for case_path in self._case_paths:
            c = os.path.basename( os.path.abspath(case_path) )
            self._cases.append(c)
            self._volumes[c] = None
            self._kidney[c] = None
            self._tumor[c] = None


def _test_setup(args):
    cfg = vars(args)
    dataset = Kidney(cfg["data_supdir"], use_coordconv=cfg["use_coordconv"],
            return_path=True)
    N = len(dataset)
    print("Created Kidney dataset, length %d" % N)
    output_dir = cfg["output_dir"]
    if pe(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return cfg,dataset,output_dir

def _test_infer(args):
    cfg,dataset,output_dir = _test_setup(args)
    raise NotImplementedError()

def _test_main(args):
    cfg,dataset,output_dir = _test_setup(args)
    num_input_ch = 4 if cfg["use_coordconv"] else 1
    for index in range(cfg["num_outputs"]):
        vol,label,return_path = dataset[index]
        title = os.path.splitext( os.path.basename(return_path) )[0]
        path = pj(output_dir, "%d_%s.png" % (index, title))
        # We treat depth*num_input_ch as batch size
        vol = vol.view((-1, 1, *vol.shape[2:]))
        label = label.view((-1, 1, *label.shape[2:]))
        print(vol.shape, label.shape)
        tv.utils.save_image(vol, path)
        label_path = pj(output_dir, "%d_%s_label.png" % (index, title))
        tv.utils.save_image(label, label_path)
        print("%d. Wrote %s, shape: %s" % (index, path, repr(vol.shape)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/Kidney/Volumes"))
    parser.add_argument("-t", "--test", type=str, default="main",
            choices=["main", "inference"])
    parser.add_argument("--no-cc", "--no-coordconv", dest="use_coordconv",
            action="store_false")
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Training/volume/test_out/Kidney"))
    parser.add_argument("-n", "--num-outputs", type=int, default=10)
    args = parser.parse_args()
    if args.test=="main":
        _test_main(args)
    else:
        _test_infer(args)

