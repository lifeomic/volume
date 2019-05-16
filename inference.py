"""
Run inference on volumetric datasets
"""

import argparse
import csv
import cv2
import numpy as np
import os
import shutil
import sys
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFilter, ImageChops

from scipy.stats import norm

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader

from general.utils import read_session_config
from pytorch.pyt_utils.utils import get_recent_model, get_resnet_model

from lo_utils.utils import make_xyzr_gradients

from datasets.pancreas import get_Pancreas_loaders, Pancreas
from models.vnet import get_vnet, VNet

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def _draw_anno(img_path, anno_path, pred_img):
    pred_img = Image.fromarray(pred_img)
    pred_img = pred_img.resize((512,512))
    edges = pred_img.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter( ImageFilter.MaxFilter(3) )
    edges = edges.convert("RGB")
    img = Image.open(img_path)
    img = img.resize((512,512))
    anno = ImageChops.lighter(img, edges)
    anno.save(anno_path)

# Returns the coordinates of the covering patch set for a given volume
def _get_covering_patches(index, dataset, cfg):
    dp,ht,wd = dataset.get_shape(index)
    patch_dp = cfg["patch_depth"]
    patch_ht = cfg["patch_height"]
    patch_wd = cfg["patch_width"]
    patches = []
    print("Volume shape: %d, %d, %d" % (dp, ht, wd))
    print("Patch shape: %d, %d, %d" % (patch_dp, patch_ht, patch_wd))
    k = 0
    while k == 0 or k <= dp-patch_dp:
        j = 0
        while j <= ht-patch_ht:
            i = 0
            while i <= wd-patch_wd:
                p0 = (k,j,i)
                p1 = ( k+patch_dp, j+patch_ht, i+patch_wd )
                patches.append( (p0,p1) )
                i += patch_wd
            j += patch_ht
        k += patch_dp
    return patches

def _get_full_pred(index, dataset, model, cfg):
    cudev = cfg["cuda"]
    patch_coords = _get_covering_patches(index, dataset, cfg)
#    for p in patch_coords:
#        print(p)
    patch_preds = []
    for p in patch_coords:
#        print(p)
        patch = dataset.get_patch(index, p[0][::-1], p[1][::-1])
        patch = patch.view((-1, *patch.shape))
        if cudev >= 0:
            patch = patch.cuda(cudev)
#        print(patch.shape)
        patch_pred = model(patch)
#        print(torch.min(patch_pred), torch.max(patch_pred),
#                torch.median(patch_pred))
        patch_pred = torch.sigmoid(patch_pred).detach()
#        print("\t", torch.min(patch_pred), torch.max(patch_pred),
#                torch.median(patch_pred))
#        print("\t", patch_pred.shape)
        patch_preds.append(patch_pred)
    pred = _merge_patch_preds(patch_preds, patch_coords).detach()
    return pred

# Make the radial weighting mask determining the weight of the voxel predictions
def _make_patch_weight(size, sd=2.0, min_wt=0.01):
    _,_,_,grad_r = make_xyzr_gradients(size, cast_to_uint8=False)
    grad_r = sd * grad_r / 255.0
    grad_r = norm.pdf(grad_r)
    grad_r = np.minimum(grad_r, min_wt*np.ones_like(grad_r))
    return torch.FloatTensor(grad_r)

# Fuses a covering set of patches into a single volume
def _merge_patch_preds(patch_preds, patch_coords):
    cudev = patch_preds[0].device
    dp,ht,wd,patch_dp,patch_ht,patch_wd = -1,-1,-1,-1,-1,-1
    for p in patch_coords:
        if p[1][0] > dp:
            dp = p[1][0]
        if p[1][1] > ht:
            ht = p[1][1]
        if p[1][2] > wd:
            wd = p[1][2]
        if patch_dp == -1:
            patch_dp = p[1][0] - p[0][0]
        if patch_ht == -1:
            patch_ht = p[1][1] - p[0][1]
        if patch_wd == -1:
            patch_wd = p[1][2] - p[0][2]
        if patch_dp != p[1][0] - p[0][0] or patch_ht != p[1][1] - p[0][1] \
                or patch_wd != p[1][2] - p[0][2]:
            raise RuntimeError("Inconsistent patch dimensions")
    patch_weight = _make_patch_weight( (patch_dp, patch_ht, patch_wd) )
    patch_weight = patch_weight.to(cudev)
    raw_patch_vol = torch.zeros(dp,ht,wd).to(cudev)
    weights_vol = torch.zeros(dp,ht,wd).to(cudev)
#    expanded_patches = []
#    expanded_weight_masks = []
    for patch,pco in zip(patch_preds, patch_coords):
        (z0,y0,x0),(z1,y1,x1) = pco
        expat = torch.zeros(dp,ht,wd).to(cudev)
        expat[ z0:z1, y0:y1, x0:x1 ] = patch*patch_weight
        raw_patch_vol += expat
#        expanded_patches.append(expat)

        wt_mask = torch.zeros(dp,ht,wd).to(cudev)
        wt_mask[ z0:z1, y0:y1, x0:x1 ] = patch_weight
#        expanded_weight_masks.append(wt_mask)
        weights_vol += wt_mask

#    for p,w in zip(expanded_patches, expanded_weight_masks):
#        raw_patch_vol += p
#        weights_vol += w

    pred_volume = raw_patch_vol / weights_vol
    return pred_volume
    

def _write_pred_image(pred, pred_dir, img_path, resize_before_save=False):
    img_stub = os.path.splitext( os.path.basename(img_path) )[0]
    if "Pancreas" in img_path:
        case = os.path.splitext( os.path.basename( os.path.dirname( \
                os.path.dirname(img_path) ) ) )[0]
        img_stub = case + "_" + img_stub
    img = Image.open(img_path)
    wd,ht = img.size
    if len( pred.shape ) == 3:
        pred = pred.view((1, *pred.shape))
    else:
        pred = pred.view((1, 1, *pred.shape))
    pred = F.interpolate(pred, size=(ht,wd), mode="bilinear")

    # Do this two-step method because the UNet just produces logits
    # which save_image understands
    pred_path = pj(pred_dir, "%s.bmp" % img_stub)
    tv.utils.save_image(pred, pred_path)
    pred_img = Image.open(pred_path)
    if resize_before_save:
        pred_img = pred_img.resize((512,512))
    binarizer = lambda x : 255 if x > 127 else 0
    pred_img = pred_img.convert("L").point(binarizer, mode="1")
    pred_img.save(pred_path)
    return img_stub, img, pred_path, pred_img
    
def load_vnet(cfg):
    model = get_vnet(cfg)
    model.load_state_dict( torch.load(cfg["model_path"]) )
    model.eval()
    model.train = False
    return model

def seg_tile_inference(model, data_loader, cfg, epoch=-1, num_out=20):
    cudev = cfg["cuda"]
    print("Running tiled segmentation inference now, data set has %d items..." \
            % len(data_loader.dataset))

    cat_tokens = cfg[ cfg["dataset"] + "_output_cat" ].split("_")
    seg_dirs = []
    annos_dirs = []
    for i,cat in enumerate(cat_tokens):
        if epoch>=0:
            output_dir = pj(cfg["session_dir"], "images")
            seg_dir = pj(output_dir, cat, "epoch_%03d" % epoch)
            annos_dir = pj(output_dir, cat+"_annos", "epoch_%03d" % epoch)
        else:
            output_dir = cfg["output_dir"]
            seg_dir = pj(output_dir, cat)
            annos_dir = pj(output_dir, cat+"_annos")
        for d in [seg_dir, annos_dir]:
            if pe(d):
                shutil.rmtree(d)
            os.makedirs(d)
        seg_dirs.append(seg_dir)
        annos_dirs.append(annos_dir)

    length=0
    dataset = data_loader.dataset
    pred_stacks = []
    img_paths = []
    indexes = []
    annos_dir = annos_dirs[0] # TODO
    for index in range( len(dataset) ):
        if index != 5: continue
        print("\tRunning inference on volume %d..." % index)
        torch.cuda.empty_cache()
        pred = _get_full_pred(index, dataset, model, cfg)
        print("Prediction:", pred.shape)
#        pred_stacks.append(pred)
        img_paths.append( dataset.get_path(index) )

#        length += pred_stacks.size(0)
        length += 1
        indexes.append(index)

        pred = pred.view((-1, 1, *pred.shape[1:]))
#        print(torch.min(pred), torch.max(pred), torch.median(pred))
        pred[ pred<0.5 ] = 0.0
        pred[ pred>=0.5 ] = 1.0
#        print(torch.sum(pred==0.0), torch.sum(pred==1.0), torch.mean(pred))
#        print(pred.shape)
        tv.utils.save_image(pred, pj(annos_dir, "%03d.png" % index))

        print("\t...Done")
        if epoch>=0 and length==num_out:
            break
#    for pred_stack in pred_stacks: # This is supposed to handle multi-class
#        for p,(d,anno_d) in enumerate( zip(seg_dirs, annos_dirs) ):
#            preds = pred_stack[p,:,:]
#            preds = preds.view((1, *preds.shape))
#            _write_binary_preds(preds, img_paths, d, anno_d)
#
    print("...Done.  %d Volumetric predictions written to %s" \
            % (length, annos_dir))

def set_up_config(cfg):
    logfile = pj( cfg["source_session_dir"], "session.log" )
    source_cfg = read_session_config(logfile)
    cfg["model_path"] = get_recent_model( pj(source_cfg["session_dir"],
        "models") )
    for k in ["dataset", "palm_data_supdir", "pancreas_data_supdir",
            "pancreas_output_cat", "input_size",
            "use_coordconv", "patch_depth", "patch_height", "patch_width",
            "model", "num_base_chans", "num_unet_layers", "broadcast_si",
            "structured_inputs", "decoder_type"]:
        if k not in source_cfg:
            print("Warning, key %s is missing.  This is okay if running " \
                    "inference on an older dataset" % k)
            cfg[k] = None
            continue
        try:
            v = int( source_cfg[k] )
        except ValueError:
            try:
                v = float( source_cfg[k] )
            except ValueError:
                v = source_cfg[k]
        cfg[k] = v
    if len( cfg["output_dir"] ) == 0:
        cfg["output_dir"] = pj(source_cfg["session_dir"], "eval")
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    if cfg["dataset_mode"] == "train":
        cfg["train_valid_split"] = 1.0
    else:
        cfg["train_valid_split"] = 0.0        
    cfg["output_path"] = ""
    print("Configuration:")
    devnull = [print("%s: %s" % (k,v)) for k,v in cfg.items()]
    return cfg

def main(args):
    cfg = vars(args)
    cfg = set_up_config(cfg)
    if cfg["dataset"]=="pancreas":
        data_loader = get_Pancreas_loaders(cfg, loaders="test")
    else:
        raise RuntimeError("Unrecognized dataset, %s" % cfg["dataset"])
    model = load_vnet(cfg)
    seg_tile_inference(model, data_loader, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-session-dir", "--ssd", type=str,
            dest="source_session_dir", required=True)
    parser.add_argument("--output-dir", type=str, default="",
            help="If left empty, output directory taken from the model path")
    parser.add_argument("--mode", "--dataset-mode", dest="dataset_mode",
            type=str, default="test", choices=["train", "valid", "test"])

    # Note these are for reentrant inputs, will be different for inference
    # than for training, thus cannot be inferred from the source session dir
    parser.add_argument("--ecd", "--extra-channel-dir",
            dest="extra_channel_dir", type=str, default=None)
    parser.add_argument("--ecd2", "--extra-channel-dir2",
            dest="extra_channel_dir2", type=str, default=None)

    parser.add_argument("-m", "--model-name", type=str,
            help="If empty, most recent model will be chosen")
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--cuda", type=int, default=0, help="-1 for CPU")

    args = parser.parse_args()
    main(args)

