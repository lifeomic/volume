"""
Should only be run once, a script to generate .png images and corresponding
segmentation maps from the KITS2019 challenge data
"""

import argparse
import numpy as np
import os
import shutil
from PIL import Image

from starter_code.utils import load_case
from starter_code.visualize import hu_to_grayscale, DEFAULT_HU_MIN, \
        DEFAULT_HU_MAX

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(args):
    cfg = vars(args)
    sd = cfg["source_dir"]
    cases = [pj(sd,d) for d in os.listdir(sd) if d.startswith("case_")]
    print("%d cases found" % len(cases))
    if cfg["volumes"]:
        supsupdir = os.path.dirname( os.path.abspath( cfg["data_supdir"] ) )
        volumes_supdir = pj(supsupdir, "Volumes")
        if pe(volumes_supdir):
            shutil.rmtree(volumes_supdir)
        os.makedirs(volumes_supdir)
        for d in ["imgs", "kidney", "tumor"]:
            os.makedirs(pj(volumes_supdir, d))
    for case in cases:
        case_name = os.path.basename(os.path.abspath(case))
        print("Processing case %s..." % case_name)
        vol,seg = load_case(case)
        if not cfg["volumes"]:
            new_dir = pj(cfg["data_supdir"], case_name)
            if pe(new_dir):
                shutil.rmtree(new_dir)
            os.makedirs(new_dir)
            for d in ["imgs", "kidney", "tumor"]:
                os.makedirs(pj(new_dir, d))
        vol = vol.get_data()
        vol = hu_to_grayscale(vol, DEFAULT_HU_MIN, DEFAULT_HU_MAX)\
                .astype(np.uint8)
        seg = seg.get_data()
        if cfg["volumes"]:
            if len(vol.shape)==4:
                vol = vol[:,:,:,0]
            assert(len(vol.shape) == 3)
            save_str = pj(volumes_supdir, "%s", case_name)
            np.save(save_str % "imgs", vol, allow_pickle=True)
            panc = 255 * (seg!=1).astype(np.uint8)
            np.save(save_str % "kidney", panc, allow_pickle=True)
            tumor = 255 * (seg!=2).astype(np.uint8)
            np.save(save_str % "tumor", tumor, allow_pickle=True)
        else:
            ct = 0
            for img,label in zip(vol,seg):
                img = Image.fromarray(img).convert("L")
                img.save( pj(new_dir, "imgs/%04d.png" % ct) )
                panc = 255 * (label!=1).astype(np.uint8)
                panc = Image.fromarray(panc).convert("L")
                panc.save( pj(new_dir, "kidney/%04d.png" % ct) )
                tumor = 255 * (label!=2).astype(np.uint8)
                tumor = Image.fromarray(tumor).convert("L")
                tumor.save( pj(new_dir, "tumor/%04d.png" % ct) )
                ct += 1
    print("\t... Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str, 
            default=pj(HOME, "Datasets/BiomedVolumes/KiTS19/Cases"))
    parser.add_argument("--source-dir", type=str,
            default=pj(HOME, "Repos/neheller/kits19/data"))
    parser.add_argument("-v", "--volumes", action="store_true")
    args = parser.parse_args()
    main(args)

