import yaml
import math
import os
import timm
import logging
import warnings
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
from saver import Saver, resume
from utils import *
from optimzier import *
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
        '--dataroot',
        default="./datasets/MSRS/",
        type=str,
        help="Path to save training(test) data"
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
        '--nThreads',
        default=8,
        type=int,
        help="Dataloader config: num_workers"
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
    parser.add_argument(
        '--deterministic',
        default=False,
        type=bool,
        help="keeping conv ops stable"
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help="Which gpu id to use"
    )
    parser.add_argument(
        '--resume',
        default=None，
        type=str,
        help="Continue resume training"
    )
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    
    args = parser.parse_args()
    # assert args.config is not None
    # cfg = load_cfg_from_cfg_file(args.config)
    cfg = args
    return cfg

@logger.catch
def main():
    cfg = get_parser()
    print(f"==== Config ===== \n {cfg}")
    #  random seed
    seed = np.random.randint(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # define model, optimzier, scheduler 
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    # Fusion_model = FusionModel(cfg).to(device)  ## ToDO
    # define dataset
    train_dataset = MSRSData(cfg, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nThreads,
        shuffle=True
    ) ## [ir, vis, lable, bi, bd，mask]
    test_dataset = MSRSData(cfg, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers = cfg.nThreads,
        shuffle=False)   

    ## 先加载dataloader 计算每个epoch的的迭代步数 然后计算总的迭代步数
    ep_iter = len(train_loader)
    max_iter = cfg.num_epoch * ep_iter
    ## test 
    # print(f"==== TEST ep_iter:{ep_iter} \n train_list_len:{len(next(iter(train_loader)))} \ntrain_img.shape:{next(iter(train_loader))[0].shape}")
    print('Training iter: {}'.format(max_iter))
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3
    # max_iter = 150000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optimizer = Optimizer(
            model = MPF_model,  ## TODO
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    if args.resume:
        FusionModel, optimizer.optim, ep, total_ir = resume

if __name__ == '__main__':
    main()