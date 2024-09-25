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
                self.model = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-base-coco"
                )
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                # pixel level module contains both the backbone and the pixel decoder
                for param in self.model.model.pixel_level_module.parameters():
                        param.requires_grad = False

                # Confirm that the parameters are correctly frozen
                for name, param in self.model.model.pixel_level_module.named_parameters():
                        assert not param.requires_grad

                print(f"> Initialized {self.__class__.__name__} with {kwargs}")

        def forward(self, input):
                input = input.repeat(1, 3, 1, 1)
                inputs = self.processor(images=input, return_tensors="pt").to(self.device)
                outputs = self.model(pixel_values=inputs.pixel_values, class_labels = range(5))
                size = input.size()
                print(outputs.masks_queries_logits.size())
                predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(size[2], size[3]) for _ in range(size[0])]
                )
                print(predicted_semantic_map.size())
                return torch.stack(predicted_semantic_map, dim=0).unsqueeze(1)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
