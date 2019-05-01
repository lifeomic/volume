"""
Volumetric segmentation for the Pancreas dataset
"""

import argparse
import csv
import numpy as np
import os
import shutil
import sys
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from general.utils import copy_code, get_project_dir, \
        make_or_get_session_dir, plot_confusion_matrix, retain_session_dir, \
        write_training_results, write_arguments, write_parameters
from pytorch.pyt_utils.utils import get_summary_writer, save_model_pop_old
from lo_utils.utils import weighted_bce_loss, weighted_dice

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.pancreas import Pancreas

from models.vnet import VNet

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def _union_mask(x, y, threshold=0.5):
    return (x<threshold) | (y<threshold)

# LeeJunHyun
def get_sensitivity(SR,GT,threshold=0.5,is_dark_fgnd=False):
    # Sensitivity == Recall
    SR = SR < threshold if is_dark_fgnd else SR > threshold
    GT = GT == torch.min(GT) if is_dark_fgnd else GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)

    return SE

# LeeJunHyun
def get_precision(SR,GT,threshold=0.5,is_dark_fgnd=False):
    SR = SR < threshold if is_dark_fgnd else SR > threshold
    GT = GT == torch.min(GT) if is_dark_fgnd else GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

# LeeJunHyun
def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return F1

def get_Dice(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold, is_dark_fgnd=True)
    PC = get_precision(SR,GT,threshold=threshold, is_dark_fgnd=True)
    Dice = 2*SE*PC/(SE+PC + 1e-6)
    return Dice

def get_iou(preds, gt, threshold=0.5):
    union = torch.sum( _union_mask(preds, gt, threshold=threshold) ).float()
    intersection = torch.sum( (preds<threshold) & (gt<threshold) ).float()
    iou = intersection / (union + 1e-6)
    return iou

def get_Pancreas_loaders(cfg):
    ht,wd,dp = cfg["patch_depth"], cfg["patch_height"], cfg["patch_width"]
    ds_kwargs = { "data_supdir" : cfg["pancreas_data_supdir"],
            "output_cat" : cfg["pancreas_output_cat"],
            "patch_size" : (ht,wd,dp) }
    for k in ["use_coordconv", "train_valid_split"]:
        ds_kwargs[k] = cfg[k]
    dl_kwargs = { "num_workers" : cfg["num_workers"],
            "batch_size" : cfg["batch_size"] }
    train_dataset = Pancreas(mode="train", **ds_kwargs)
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    test_dataset = Pancreas(mode="valid", return_path=True, **ds_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)
    return train_loader,test_loader

def get_model(cfg):
    output_cat = cfg[ cfg["dataset"] + "_output_cat" ]
    num_output_ch = len( output_cat.split("_") )
    num_input_ch = 4 if cfg["use_coordconv"] else 1
    model = VNet(num_input_ch=num_input_ch)
    if cfg["use_coordconv"]:
        print("Built VNet model with CoordConv")
    else:
        print("Built VNet model without CoordConv")
    if cfg["cuda"] >= 0:
        model = model.cuda( cfg["cuda"] )
    return model

def train(cfg, train_loader, test_loader, model, writer):
    with open( pj(cfg["session_dir"], "session.log"), "a" ) as fp:
        fp.write("\nMODEL:\n" + str(model) )
    if len(cfg["resume_path"]) > 0:
        model.load_state_dict( torch.load(cfg["resume_path"]) )
        model_name = os.path.splitext( os.path.basename(cfg["resume_path"]) )[0]
        num_idx = len(model_name) - model_name[::-1].index("_")
        start_epoch = int( model_name[num_idx:] )
    else:
        start_epoch = 0
    cudev = cfg["cuda"]
    num_tasks = model.get_output_ch() 
    output_cat = cfg[ cfg["dataset"] + "_output_cat" ]
    tasks = output_cat.split("_")
    if num_tasks != len(tasks):
        raise RuntimeError("Need to update training code")
    if cfg["ce_weighting"] == "class":
        weights = torch.FloatTensor( train_loader.dataset\
                .get_class_weights(min_wt=cfg["weighted_ce_min"]) )
        with open(pj(cfg["session_dir"], "session.log"), "a") as fp:
            fp.write("Class weights: %s\n" % repr(weights.data))
        if cudev>=0:
            weights = weights.cuda(cudev)
        if cfg["loss_func"] == "bce":
            criterion = lambda x,gt : weighted_bce_loss(x, gt, weights=weights)
        else:
            criterion = lambda x,gt : weighted_dice(x, gt, weights=weights)
    elif cfg["ce_weighting"] == "union":
        if cfg["loss_func"] == "bce":
            criterion = lambda x,gt : weighted_bce_loss(x, gt,
                    restrict_to_union=True)
        else:
            raise NotImplementedError()
    else:
        if cfg["loss_func"] == "bce":
            criterion = nn.BCELoss()
        else:
            raise NotImplementedError()
    images_dir = pj(cfg["session_dir"], "images")
    if not pe(images_dir):
        os.makedirs(images_dir)
    models_dir = pj(cfg["session_dir"], "models")
    if not pe(models_dir):
        os.makedirs(models_dir)
    cli = cfg["console_log_interval"]
    si = cfg["sample_interval"]
    best_test_F1 = 0
    best_test_Dice = 0
    best_test_iou = 0
    best_test_loss = np.inf
    learning_rates = [ cfg["lr"], cfg["lr"]/5, cfg["lr"]/25 ]
    epoch_drop_ct = 0
    lr_ct = 0
    best_model_path = None
    iter_ct = 0
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam( [{"params" : model.parameters()}],
                betas=(cfg["b1"], cfg["b2"]), eps=cfg["eps"],
                weight_decay=cfg["weight_decay"], lr=learning_rates[lr_ct])
    elif cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD([{"params" : model.parameters()}],
                lr=learning_rates[lr_ct], momentum=cfg["momentum"])
    else:
        raise RuntimeError("Unrecognized optimizer, %s" % cfg["optimizer"])
    last_lr = learning_rates[lr_ct]
    for epoch in range( start_epoch, cfg["max_num_epochs"] ):
        if pe(pj(cfg["session_dir"], "stop_experiment.txt")):
            with open(pj(cfg["session_dir"], "session.log"), "a") as fp:
                fp.write("Breaking out of experiment at epoch %d\n" % epoch)
            break
        train_loader.dataset.init_random_vals()
        if learning_rates[lr_ct] != last_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rates[lr_ct]
            last_lr = learning_rates[lr_ct]
        running_loss = 0.0
        running_F1 = 0.0
        running_Dice = 0.0
        running_iou = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            if pe(pj(cfg["session_dir"], "stop_train.txt")):
                with open(pj(cfg["session_dir"], "session.log"), "a") as fp:
                    fp.write("Breaking out of training at iteration %d\n" % i)
                break
            x,gt = data
            if cudev >= 0:
                x = x.cuda(cudev)
                gt = gt.cuda(cudev)

#            print(x.shape, gt.shape)
            optimizer.zero_grad()
            preds = model(x)
            preds = torch.sigmoid( preds.view((-1,1)) )
            gt = gt.view((-1,1))
#            preds = torch.sigmoid( preds.view(-1, preds.size(1)) )
#            gt = gt.view(-1, gt.size(1))
#            gt_onehot = torch.FloatTensor(preds.size())
#            if cudev>=0:
#                gt_onehot = gt_onehot.cuda(cudev)
#            gt_onehot.zero_()
#            gt_onehot.scatter_(1, gt, 1)
#            loss = criterion(preds.view(-1,1), gt_onehot.view(-1,1))
            loss = criterion( preds.view(-1, 1), gt.view(-1, gt.size(1)) )
            loss.backward()
            optimizer.step()

#            print(preds.shape, gt.shape)
            F1 = get_F1(preds, gt)
            Dice = get_Dice(preds, gt)
            iou = get_iou(preds, gt).item()
            running_loss += loss.item()
            running_F1 += F1
            running_Dice += Dice
            running_iou += iou
            if i % cli == cli-1:
                writer.add_scalars("Loss", { "train" : loss }, iter_ct)
                writer.add_scalars("F1", { "train" : F1 }, iter_ct)
                writer.add_scalars("Dice", { "train" : Dice }, iter_ct)
                writer.add_scalars("IOU", { "train" : iou }, iter_ct)
                print("[%3d,%5d] Train: loss: %.3f, F1: %.3f, Dice: %.3f, " \
                        "IOU: %.3f" % (epoch, i, running_loss/cli,
                            running_F1/cli, running_Dice/cli, running_iou/cli))
                running_loss = 0.0
                running_F1 = 0.0
                running_Dice = 0.0
                running_iou = 0.0
            iter_ct += 1

        optimizer.zero_grad()

#        if epoch % si == si-1:
#            model.train(False)
#            model.eval()
#            running_loss = 0.0
#            running_F1 = np.zeros((num_tasks,))
#            running_Dice = np.zeros((num_tasks,))
#            running_iou = np.zeros((num_tasks,))
#            ct = 0
#            for i, data in enumerate(test_loader):
#                if pe(pj(cfg["session_dir"], "stop_valid.txt")):
#                    with open(pj(cfg["session_dir"], "session.log"), "a") as fp:
#                        fp.write("Breaking out of validation at iteration %d\n"\
#                                % i)
#                    break
#                x,gt,_ = data
#                if cudev >= 0:
#                    x = x.cuda(cudev)
#                    gt = gt.cuda(cudev)
#
#                preds = torch.sigmoid( model(x) )
#                loss = criterion( preds.view(preds.size(0), -1),
#                        gt.view(gt.size(0), -1))
#                loss.backward()
#                # Bizarrely, the line above is necessary or else several GB
#                # of additional memory are consumed
#
#                running_loss += loss.item()
#                task_sz = preds.view(preds.size(0), -1).size(1) // num_tasks
#                for t in range(num_tasks):
#                    preds_view = preds.view(preds.size(0),-1)\
#                            [:, t*task_sz : (t+1)*task_sz]
#                    gt_view = gt.view(gt.size(0),-1)\
#                            [:, t*task_sz : (t+1)*task_sz]
#                    F1 = get_F1(preds_view, gt_view)
#                    Dice = get_Dice(preds_view, gt_view)
#                    iou = get_iou(preds_view, gt_view).item()
#                    running_F1[t] += F1
#                    running_Dice[t] += Dice
#                    running_iou[t] += iou 
#                ct += 1
#
#            optimizer.zero_grad()
#
#            mean_test_loss = running_loss / ct
#            print("***Mean test loss: %f" % (mean_test_loss))
#            writer.add_scalars("Loss", { "test" : mean_test_loss }, iter_ct)
#            F1_text = "***Mean test F1: "
#            Dice_text = "***Mean test Dice: "
#            iou_text = "***Mean test IOU: "
#            log_text = "Epoch %03d: Loss: %.4f" % (epoch, mean_test_loss)
#            mean_test_F1s = []
#            mean_test_Dices = []
#            mean_test_ious = []
#            for t in range(num_tasks):
#                mean_test_F1 = running_F1[t] / ct
#                mean_test_Dice = running_Dice[t] / ct
#                mean_test_iou = running_iou[t] / ct
#
#                task = tasks[t]
#                writer.add_scalars("F1",
#                        { "test/"+task : mean_test_F1 }, iter_ct)
#                writer.add_scalars("Dice",
#                        { "test/"+task : mean_test_Dice }, iter_ct)
#                writer.add_scalars("IOU",
#                        { "test/"+task : mean_test_iou }, iter_ct)
#
#                F1_text += "%s, %f; " % (task, mean_test_F1)
#                Dice_text += "%s, %f; " % (task, mean_test_Dice)
#                iou_text += "%s, %f; " % (task, mean_test_iou)
#                log_text += "\n\t%s: F1 %.4f, Dice %.4f, IOU %.4f" % (task,
#                        mean_test_F1, mean_test_Dice, mean_test_iou)
#                mean_test_F1s.append(mean_test_F1)
#                mean_test_Dices.append(mean_test_Dice)
#                mean_test_ious.append(mean_test_iou)
#            print(F1_text)
#            print(Dice_text)
#            print(iou_text)
#            with open(pj(cfg["session_dir"], "session.log"), "a") as fp:
#                fp.write(log_text + "\n")
#
#            seg_inference(model, test_loader, cfg, epoch=epoch, num_out=20,
#                    criterion=criterion)
#
#            if mean_test_loss < best_test_loss:
#                best_model_path = save_model_pop_old(model, cfg["model"], epoch,
#                        models_dir)
#                best_test_loss = mean_test_loss
#                best_test_F1 = np.mean(mean_test_F1s)
#                best_test_Dice = np.mean(mean_test_Dices)
#                best_test_iou = np.mean(mean_test_ious)
#                epoch_drop_ct = 0
#            elif epoch_drop_ct+1 == cfg["num_epochs_for_drop"]:
#                lr_ct += 1
#                if lr_ct == len(learning_rates):
#                    print("Training ended due to early stopping.")
#                    print("Session directory: %s" % (cfg["session_dir"]))
#                    break
#                print("Learning rate reduced to %f" % learning_rates[lr_ct])
#                if best_model_path is not None:
#                    sd = torch.load(best_model_path)
#                    model.load_state_dict(sd)
#                epoch_drop_ct = 0
#            else:
#                epoch_drop_ct += 1
#
#            retain_session_dir( cfg["session_dir"] )

        torch.cuda.empty_cache()

    results_dict = OrderedDict()
    results_dict["loss"] = best_test_loss
    results_dict["F1"] = best_test_F1
    results_dict["Dice"] = best_test_Dice
    results_dict["IOU"] = best_test_iou
    results_dict["iter_ct"] = iter_ct
    results_dict["end_lr"] = learning_rates[-1]
    write_training_results(results_dict, cfg, os.path.dirname(\
            cfg["sessions_supdir"]), "trainer")

    seg_inference(model, test_loader, cfg, epoch=epoch, num_out=-1,
            criterion=criterion)

def main(args):
    cfg = vars(args)
    for k,v in cfg.items():
        if v=="none":
            cfg[k] = None
    output_cat = cfg[ cfg["dataset"] + "_output_cat" ]
    cfg["session_dir"] = make_or_get_session_dir(cfg["sessions_supdir"],
            cfg["model"], cfg["dataset"] + "_" + output_cat, cfg["resume_path"])
    print("Session directory: %s" % cfg["session_dir"])
    write_arguments(cfg["session_dir"])
    write_parameters(cfg)
    copy_code( get_project_dir("volume", os.path.abspath(__file__)),
            cfg["session_dir"] )
    writer = get_summary_writer(cfg["session_dir"])
    if cfg["dataset"] == "pancreas":
        train_loader,test_loader = get_Pancreas_loaders(cfg)
    else:
        raise RuntimeError("Unrecognized dataset, %s" % cfg["dataset"])
    print("Length of training dataset, %s: %d" \
            % (train_loader.dataset.get_name(), len(train_loader.dataset)))
    print("Length of validation dataset, %s: %d" \
            % (test_loader.dataset.get_name(), len(test_loader.dataset)))
    model = get_model(cfg)
    cfg["dataset_mode"] = "train" # This is for inference
    train(cfg, train_loader, test_loader, model, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-output-dir", type=str,
            default=pj(HOME, "Training/volume/test_out/trainer"))

    # Dataset
    parser.add_argument("-d", "--dataset", type=str, default="pancreas",
            choices=["pancreas"])
    parser.add_argument("-m", "--model", type=str, default="vnet")
    parser.add_argument("--train-valid-split", type=float, default=0.85)
    parser.add_argument("--pancreas-data-supdir", type=str,
            default=pj(HOME, "Datasets/Pancreas/Volumes"))
    parser.add_argument("--pancreas-output-cat", type=str, default="pancreas")

    # Augmentation
    parser.add_argument("--crop-pct", type=float, default=0.75)
    parser.add_argument("--no-coordconv", dest="use_coordconv",
            action="store_false")

    # Model
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--patch-depth", type=int, default=64)
    parser.add_argument("--patch-height", type=int, default=128)
    parser.add_argument("--patch-width", type=int, default=128)
    parser.add_argument("--sessions-supdir", type=str,
            default=pj(HOME, "Training/volume/sessions"))
    parser.add_argument("--resume-path", type=str, default="",
            help="Path to model to resume training.  Should be in session_<n>/"\
                    "models")

    # Optimizer
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
            default=0.0001)
    parser.add_argument("--num-epochs-for-drop", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="Adam",
            choices=["Adam", "SGD"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--b1", type=float, default=0.5,
            help="Adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
            help="Adam: decay of first order momentum of gradient")
    parser.add_argument("--eps", type=float, default=1e-8,
            help="Adam: Term added for numerical stability")
    parser.add_argument("--weight-decay", type=float, default=0.0,
            help="Adam: weight decay (L2 penalty)")

    # Loss function and weighting
    parser.add_argument("--loss-func", type=str, default="bce",
            choices=["bce", "dice"])
    parser.add_argument("--ce-weighting", type=str, default="class",
            choices=["none", "class"],
            help="Weight class entropy calculation by class size")
    parser.add_argument("--weighted-ce-min", type=float, default=1.0,
            help="Minimum weight that large categories can have")

    parser.add_argument("--max-num-epochs", type=int, default=1000)
    parser.add_argument("--console-log-interval", type=int, default=10,
            help="Measured in iterations")
    parser.add_argument("--sample-interval", type=int, default=1,
            help="Measured in epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--cuda", type=int, default=0, help="-1 for CPU")

    args = parser.parse_args()
    main(args)

