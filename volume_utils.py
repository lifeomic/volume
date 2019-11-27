"""
Various utilities
"""


import numpy as np
import os
import torch

from PIL import Image, ImageDraw, ImageChops

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def _get_weights_vec(x, target, weights=None):
    if weights is not None:
        wvec = torch.FloatTensor(*x.shape).to( x.device )
        num_fgnd = len(weights) - 1
        block_sz = wvec.shape[1] // num_fgnd
        for i,w in enumerate(weights[:-1]):
            wvec_view = wvec[:, i*block_sz : (i+1)*block_sz]
            target_view = target[:, i*block_sz : (i+1)*block_sz]
            wvec_view[ target_view==0 ] = w
            wvec_view[ target_view==1 ] = weights[-1]
        N = torch.ones(x.shape[0]).to(x.device) * x.shape[1]
    else:
        wvec = torch.ones(*x.shape).to( x.device )
        N = torch.ones(x.shape[0]).to(x.device) * x.shape[1]
    return wvec,N

# Inputs
#   size: (width, height)
#   do_invert: If true, will invert the pixel intensity ('photo negative')
#   do_return_numpy: Return the [0, 255] numpy arrays not PIL images
# Output
#   Returns a pair of linear gradients, the first which increases from 0 to 255
#       along the x axis, the second which does the same (top to bottom) along
#       the y axis.  They are returned as PIL images, mode L, by default
def make_xy_gradients(size, do_invert=False, do_return_numpy=False):
    wd,ht = size
    x = np.linspace(0.0, 255.0, wd)
    y = np.linspace(0.0, 255.0, ht)
    if do_invert:
        x = 255.0 - x
        y = 255.0 - y
    x = np.tile(x, (ht, 1))
    y = np.transpose( np.reshape( np.tile(y, (1, wd)), (wd, ht)) )
    if do_return_numpy:
        return x,y
    return Image.fromarray(x.astype(np.uint8)).convert("L"),\
            Image.fromarray(y.astype(np.uint8)).convert("L")

# Inputs
#   size: (width, height)
#   do_invert: If true, will invert the pixel intensity ('photo negative')
# Output
#   Returns the pair of linear gradients, along with a radial gradient.  They
#   are returned as PIL images, mode L
def make_xyr_gradients(size, do_invert=False):
    x,y = make_xy_gradients(size, do_invert, do_return_numpy=True)
    x_centered = np.abs(2*x - 255.0)
    y_centered = np.abs(2*y - 255.0)
    r = np.sqrt( x_centered**2 + y_centered**2 ).clip(0, 255.0)
    return Image.fromarray(x.astype(np.uint8)).convert("L"),\
            Image.fromarray(y.astype(np.uint8)).convert("L"),\
            Image.fromarray(r.astype(np.uint8)).convert("L")

# Inputs
#   size: (depth, height, width)
# Output
#   Returns a triple of linear gradients, the first which increases from 0 to
#       1.0 along the x axis, the second which does the same (top to bottom)
#       along the y axis.  These are returned as numpy arrays.
def make_xyz_gradients(size, cast_to_uint8=True):
    if type(size)==int:
        size = (size, size, size)
    dp,ht,wd = size
    x = np.linspace(0.0, 255.0, wd)
    y = np.linspace(0.0, 255.0, ht)
    z = np.linspace(0.0, 255.0, dp)
    x = np.tile(x, (dp,ht,1))
    y = np.transpose( np.reshape( np.tile(y, (1, wd)), (wd, ht)) )
    y = np.tile(y, (dp,1,1))
    z = np.expand_dims( np.expand_dims(z, 1), 2 )
    z = np.ones((dp,ht,wd)) * z
    if cast_to_uint8:
        return x.astype(np.uint8),y.astype(np.uint8),z.astype(np.uint8)
    return x,y,z

# Inputs
#   size: (depth, height, width)
# Output
#   Returns a quadruple of linear gradients, the first which increases from 0 to
#       1.0 along the x axis, the second which does the same (top to bottom)
#       along the y axis.  The last one provides a radial gradient which is 0
#       at the center and increases.  These are returned as numpy arrays.
def make_xyzr_gradients(size, cast_to_uint8=True):
    x,y,z = make_xyz_gradients(size, cast_to_uint8=False)
    x_centered = np.abs(2*x - 255.0)
    y_centered = np.abs(2*y - 255.0)
    z_centered = np.abs(2*z - 255.0)
    r = np.sqrt( x_centered**2 + y_centered**2 + z_centered**2 ).clip(0, 255.0)
    if cast_to_uint8:
        return x.astype(np.uint8),y.astype(np.uint8),z.astype(np.uint8),\
                r.astype(np.uint8)
    return x,y,z,r

# Inputs
#   x: BxN tensor of class prediction probabilities
#   target: BxN integer tensor of class assignments
# Output
#   Binary cross entropy, weighted by the prevalence in the image of each
#       category
def weighted_bce_loss(x, target, weights=None, restrict_to_union=False,
        union_thresh=0.5):
    if x.shape != target.shape:
        raise RuntimeError("Different input and target shapes not supported, " \
                "%s vs. %s" % (x.shape, target.shape))
    if len(x.shape)!=2:
        raise RuntimeError("Expecting 2D tensor at this point, got %s" \
                % repr(x.shape))
    EPS = 1e-6
    if restrict_to_union:
        if weights is not None:
            raise RuntimeError("If restrict_to_union is set, no weights should"\
                    " be supplied")
        wvec = _union_mask(x, target, threshold=union_thresh).float()
        N = torch.sum(wvec, dim=1) + 1
    else:
        wvec,N = _get_weights_vec(x, target, weights)
    bce = -wvec * (target*torch.log(x+EPS) + (1.0-target)*torch.log(1.0-x+EPS))
    return torch.mean( torch.sum(bce, dim=1) / N )

# Inputs
#   x: BxN tensor of class prediction probabilities
#   target: BxN integer tensor of class assignments
# Output
#   Dice score, weighted by the prevalence in the image of each category
def weighted_dice(x, target, weights=None):
    if x.shape != target.shape:
        raise RuntimeError("Different input and target shapes not supported, " \
                "%s vs. %s" % (x.shape, target.shape))
    if len(x.shape)!=2:
        raise RuntimeError("Expecting 2D tensor at this point, got %s" \
                % repr(x.shape))
    EPS = 1e-6
    weights = weights / torch.sum(weights)
    wvec,_ = _get_weights_vec(x, target, weights)
    pred = 1 - x
    gt = 1 - target
    dice = 2 * torch.sum(wvec*pred*gt, dim=1) \
           / (torch.sum(pred*pred, dim=1) + torch.sum(gt*gt, dim=1))
    return 1.0 - torch.mean(dice)


