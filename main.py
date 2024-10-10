#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import multiprocessing
import time
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
from os import environ

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from models import UNet
import utils
from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   prepare_wandb_login)

from losses import create_loss_fn

import wandb #TODO: remove all wandb instances on final submission

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}


def setup(args) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = eval(args.model)(1, K, **vars(args))
    net.init_weights()
    net.to(device)

    # lr = 0.0005 # Initial LR for ENet
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_Tmult, eta_min=1e-7)

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    if args.scratch:
        tmpdir = environ["TMPDIR"]
        root_dir = Path(tmpdir+"/data") / args.dataset
    else:
        root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug,
                             remove_unannotated=args.remove_unannotated)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug,
                           remove_unannotated=args.remove_unannotated)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return net, optimizer, scheduler, device, train_loader, val_loader, K


def runTraining(args):

    start = time.time()
    print(f">>> Setting up to train on {args.dataset} with {args.model}")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)

    loss_fn = create_loss_fn(args, K)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0
    best_metrics = {}

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val' and not args.dry_run:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # Apply LR scheduler
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']
        print("Epoch %d: AdamW lr %.4f -> %.4f" % (e, before_lr, after_lr))

        metrics = utils.save_loss_and_metrics(K, e, args.dest,
                                              loss=[log_loss_tra, log_loss_val],
                                              dice=[log_dice_tra, log_dice_val])
        wandb.log(metrics)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            if not args.dry_run:
                best_folder = args.dest / "best_epoch"
                if best_folder.exists():
                        rmtree(best_folder)
                copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            best_metrics = metrics

    for key, value in best_metrics.items():
        wandb.run.summary[key] = value
    end = time.time()
    print(f"[FINISHED] Duration: {(end - start):0.2f} s")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='SEGTHOR', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--args', default='')
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")

    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--lr_scheduler_T0', type=int, default=10, help="T0 for the LR scheduler")
    parser.add_argument('--lr_scheduler_Tmult', type=int, default=2, help="Tmult for the LR scheduler")

    parser.add_argument('--dropoutRate', type=float, default=0.2, help="Dropout rate for the ENet model")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter for loss functions")
    parser.add_argument('--beta', type=float, default=0.5, help="Beta parameter for loss functions")
    parser.add_argument('--focal_alpha', type=float, default=0.25, help="Alpha parameter for Focal Loss")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help="Gamma parameter for Focal Loss")

    # Optimize snellius batch job
    parser.add_argument('--scratch', action='store_true', help="Use the scratch folder of snellius")
    parser.add_argument('--dry_run', action='store_true', help="Disable saving the image validation results on every epoch")

    # Arguments for more flexibility of the run
    parser.add_argument('--remove_unannotated', action='store_true', help="Remove the unannotated images")
    parser.add_argument('--loss', default='CrossEntropy', choices=['CrossEntropy', 'Dice', 'FocalLoss', 'CombinedLoss', 'FocalDiceLoss', 'TverskyLoss'])
    parser.add_argument('--model', type=str, default='ENet', choices=['ENet', 'shallowCNN', 'UNet'])
    parser.add_argument('--run_prefix', type=str, default='', help='Name to prepend to the run name')
    parser.add_argument('--run_group', type=str, default=None, help='Your name so that the run can be grouped by it')

    # Arguments for running with different backbones
    parser.add_argument('--encoder_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--unfreeze_enc_last_n_layers', type=int, default=0, help="Train the last n layers of the encoder")

    args = parser.parse_args()
    prefix = args.run_prefix + '_' if args.run_prefix else ''
    lr = f'lr({"{:.0E}".format(args.lr)})_' if args.lr != 0.0005 else ''
    unfreeze_num_layers = f'(unfreeze-{args.unfreeze_enc_last_n_layers})' if args.unfreeze_enc_last_n_layers != 0 else ''
    run_name = f'{prefix}{lr}{args.loss}_{args.model}_{args.encoder_name}{unfreeze_num_layers}'
    run_name = 'DEBUG_' + run_name if args.debug else run_name
    args.dest = args.dest / run_name

    # Added since for python 3.8+, OS X multiprocessing starts processes with spawn instead of fork
    # see https://github.com/pytest-dev/pytest-flask/issues/104
    multiprocessing.set_start_method("fork")

    prepare_wandb_login()
    wandb.login()
    wandb.init(
        entity="ai_4_mi",
        project="SegTHOR",
        name=run_name,
        config=vars(args),
        mode="disabled" if args.debug else "online",
        group=args.run_group
    )

    print(f">> {run_name} <<")
    pprint(args)
    runTraining(args)


if __name__ == '__main__':
    main()
