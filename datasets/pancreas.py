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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import crop_max_rect, make_xy_gradients, make_xyr_gradients, \
        make_rt_gradients, make_sinusoids

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retina_base import ClassifierBase

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


