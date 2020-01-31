"""
This is a one-off file to make the x, y, z gradients corresponding to the
KiTS19 volumes
"""

import argparse
import numpy as np
import os
import shutil
from utils import make_xyz_gradients

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(args):
    cfg = vars(args)
    vols_dir = pj(cfg["data_supdir"], "imgs")
    grad_dirs = [pj(cfg["data_supdir"],g) for g in ["grad_x","grad_y","grad_z"]]
    for grad_dir in grad_dirs:
        if pe(grad_dir):
            shutil.rmtree(grad_dir)
        os.makedirs(grad_dir)
    for vol_name in os.listdir(vols_dir):
        vol = np.load( pj(vols_dir,vol_name), mmap_mode="r" )
        gradients = make_xyz_gradients(vol.shape)
        for g,d in zip(gradients, grad_dirs):
            np.save(pj(d, vol_name), g, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/BiomedVolumes/KiTS19/Volumes"))
    args = parser.parse_args()
    main(args)

