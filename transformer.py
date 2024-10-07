#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2024 Hoel Kervadec, Jose Dolz

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

# import json
import torch
from torch import nn
# import requests
# import evaluate
# import numpy as np
# from PIL import Image
# import albumentations as A
# from tqdm.auto import tqdm
# import matplotlib.pyplot as plt
# from typing import Tuple, Any
# from dataclasses import dataclass
# from datasets import load_dataset
# import matplotlib.patches as mpatches
# from torch.utils.data import Dataset, DataLoader
from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

torch.manual_seed(42)


def random_weights_init(m):
        pass
        # # pixel level module contains both the backbone and the pixel decoder
        # for param in m.model.pixel_level_module.parameters():
        #         param.requires_grad = False

        # # Confirm that the parameters are correctly frozen
        # for name, param in m.model.pixel_level_module.named_parameters():
        #         assert not param.requires_grad

class MaskFormer(nn.Module):
        def __init__(self, **kwargs):
                super().__init__()
                # self.processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
                self.processor = MaskFormerImageProcessor(
                ignore_index=0,
                do_reduce_labels=False,
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                )
                id2label = {i:i for i in range(5)}
                self.model = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-base-ade",
                num_labels = 5,
                id2label = id2label,
                ignore_mismatched_sizes=True
                )
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.class_labels = [torch.arange(5).to(self.device) for _ in range(8)]

                # pixel level module contains both the backbone and the pixel decoder
                for param in self.model.model.pixel_level_module.parameters():
                        param.requires_grad = False

                # Confirm that the parameters are correctly frozen
                for name, param in self.model.model.pixel_level_module.named_parameters():
                        assert not param.requires_grad

                print(f"> Initialized {self.__class__.__name__} with {kwargs}")

        def forward(self, input, mask):
                if input.size(1) == 1:
                        input = input.repeat(1, 3, 1, 1)
                inputs = self.processor(images=input, return_tensors="pt").to(self.device)
                mask_labels = [msk.float().to(self.device) for msk in mask]
                outputs = self.model(pixel_values=inputs.pixel_values, mask_labels=mask_labels, class_labels = self.class_labels)
                size = input.size()
                predicted_semantic_maps = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=[(size[2], size[3]) for _ in range(size[0])]
                )
                pred_segs = [torch.zeros(1, input.size(2), input.size(3)) for _ in range(input.size(0))]
                for i in range(len(predicted_semantic_maps)):
                        dict = predicted_semantic_maps[i]
                        predicted_semantic_map = dict['segmentation']
                        info = dict['segments_info']
                        if info:
                                for inf in info:
                                        pred_segs[i] = torch.cat((pred_segs[i], torch.where(predicted_semantic_map == inf['id'], inf['label_id'], 0).unsqueeze(0)), dim=0)
                        pred_segs[i] = torch.max(pred_segs[i], dim=0)[0]
                pred_segs = torch.stack(pred_segs, dim=0)
                return outputs, pred_segs

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
