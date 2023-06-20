import yaml
import math
import os
import timm
import logging
from collections import OrderedDict
from typing import Tuple, Union
from loguru import logger
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

import datasets
import diffusers
import transformers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.cross_attention import AttnProcessor
# from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from model import UNet2DConditionModel
from dataset import *
from utils import *
from tqdm import tqdm
import datetime
from time import time
warnings.filterwarnings("ignore")

# option parser
def get_parser():
    parser = argparse.ArgumentParser(
        description="DIFFusion Pytorch ir and vi fusion"
    )
    parser.add_argument(
        '--mode',
        default=True,
        type=bool,
        help="training mode"
    )
    parser.add_argument(
        '--num_epoch',
        default=10,
        type=int,
        help="Epoch for training"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch_size for trainig"
    )
    parser.add_argument(
        '--amp',
        default='32',
        type=str,
        help="Set amp if need using mixed precise"
    )
    parser.add_argument(
        '--optim',
        default='adam',
        type=str,
        help="Which optimizer using in training"
    )
    parser.add_argument(
        '--scheduler',
        default=None,
        type=str,
        help="Which lr scheduler in trainig"
    )
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg

def main():
    # parse options
    parser = 
