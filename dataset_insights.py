from dataset import SliceDataset
from pathlib import Path
from torchvision import transforms
import torch
from utils import class2one_hot
import numpy as np
from operator import itemgetter
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor()
])

gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        lambda nd: nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=5),
        itemgetter(0)  # Remove the batch dimension
    ])

root_dir = Path("data") / "SEGTHOR"
train_set = SliceDataset('train', root_dir, img_transform=transform, gt_transform=gt_transform, remove_unannotated=True)

counts = torch.zeros(4)
for image in tqdm(train_set):
    gt = image["gts"].sum(dim=(1,2))
    counts[gt[1:]>0] += 1

print(counts)