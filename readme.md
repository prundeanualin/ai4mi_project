# AI for medical imaging — Fall 2024 course project

## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.
![Segthor Overview](segthor_overview.png)

## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`;
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files;
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

### Setting up the environment
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```

For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

### Viewing the data
The data can be viewed in different ways:
- looking directly at the `.png` in the sliced folder (`data/SEGTHOR`);
- running the jupyter notebook that produces the graphs reported in the submitted paper ([see below](#report-graphs));
- using the provided "viewer" to compare segmentations ([see below](#viewing-the-results));
- opening the Nifti files from `data/segthor_train` with [3D Slicer](https://www.slicer.org/) or [ITK Snap](http://www.itksnap.org).

### Training a base network
Running a training
```
$ python main.py --help

usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--seed SEED] [--num_workers NUM_WORKERS] [--gpu] [--debug] [--evaluation] [--dropoutRate DROPOUTRATE] [--lr LR] [--lr_weight_decay LR_WEIGHT_DECAY] [--disable_lr_scheduler] [--alpha ALPHA] [--beta BETA] [--focal_alpha FOCAL_ALPHA] [--focal_gamma FOCAL_GAMMA] [--patience PATIENCE] [--scratch] [--dry_run]
               [--remove_unannotated] [--loss {CrossEntropy,DiceLoss,FocalLoss,CombinedLoss,TverskyLoss}] [--model {ENet,shallowCNN,UNet,UNetPlusPlus,DeepLabV3Plus}] [--run_prefix RUN_PREFIX] [--encoder_name {resnet18,resnet34,resnet50,resnet101,resnet152}] [--unfreeze_enc_last_n_layers UNFREEZE_ENC_LAST_N_LAYERS]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights). If in evaluation mode, then this is the directory where the results are saved.
  --seed SEED           Random seed to use for reproducibility of the experiments
  --num_workers NUM_WORKERS
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
  --evaluation          Will load the model from the dest_results and evaluate it on the validation set, with all the available metrics.
  --dropoutRate DROPOUTRATE
                        Dropout rate for the ENet model
  --lr LR               Learning rate
  --lr_weight_decay LR_WEIGHT_DECAY
                        Weight decay factor for the AdamW optimizer
  --disable_lr_scheduler
                        Disable OneCycle learning rate scheduler
  --alpha ALPHA         Alpha parameter for loss functions
  --beta BETA           Beta parameter for loss functions
  --focal_alpha FOCAL_ALPHA
                        Alpha parameter for Focal Loss
  --focal_gamma FOCAL_GAMMA
                        Gamma parameter for Focal Loss
  --patience PATIENCE   Patience for early stopping
  --scratch             Use the scratch folder of snellius
  --dry_run             Disable saving the image validation results on every epoch
  --remove_unannotated  Remove the unannotated images
  --loss {CrossEntropy,DiceLoss,FocalLoss,CombinedLoss,TverskyLoss}
  --model {ENet,shallowCNN,UNet,UNetPlusPlus,DeepLabV3Plus}
  --run_prefix RUN_PREFIX
                        Name to prepend to the run name
  --encoder_name {resnet18,resnet34,resnet50,resnet101,resnet152}
  --unfreeze_enc_last_n_layers UNFREEZE_ENC_LAST_N_LAYERS
                        Train the last n layers of the encoder



$ python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

The codebase uses a lot of assertions for control and self-documentation, they can easily be disabled with the `-O` option (for faster training) once everything is known to be correct (for instance run the previous command for 1/2 epochs, then kill it and relaunch it):
```
$ python -O main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

### Replicating the results from the report

In order to obtain the same results we used in our report, the following training commands should be executed:
- For Baseline-ENet:
```
$ python -O main.py --gpu --dest results/segthor --epochs 200 --dropoutRate 0.1 --lr_weight_decay 0.01 --disable_lr_scheduler --model ENet
```
All of the experiments running with the Baseline-ENet (study on different loss functions, data augmentations) use the command above as a starter and then add additional flags as needed.

- For Best-ENet:
```
$ python -O main.py --gpu --dest results/segthor --model ENet --loss CombinedLoss
```
- For UNet
```
$ python -O main.py --gpu --dest results/segthor --model UNet --loss CombinedLoss
```

- For UNetPlusPlus
```
$ python -O main.py --gpu --dest results/segthor --model UNetPlusPlus --loss CombinedLoss
```

- For DeepLabV3Plus
```
$ python -O main.py --gpu --dest results/segthor --model DeepLabV3Plus --loss CombinedLoss
```

TIP: The parameters where the default value has been used are intentionally omitted from the commands above.

### Evaluating a network

Running the evaluation of a saved model on the validation set
```
$ python main.py --gpu --evaluation --dataset <chosen_dataset> --dest <checkpoint_path> --model <chosen_model>
```
Where:
- `chosen_dataset` - decides the dataset to run the evaluation on
- `checkpoint_path` - path from the project root to the folder that contains the saved checkpoint of the model.
- `chosen_model` - the model to load from the checkpoint file

### Viewing the results
#### Report graphs
In order to generate the graphs used in the report, you should run the `plot.ipynb` jupyter notebook.

#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back):
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


```
$ python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
    --legend --class_names background esophagus heart trachea aorta
```
![Example of the viewer on SegTHOR](viewer_segthor.png)

#### 3D viewers
To look at the results in 3D, it is necessary to reconstruct the 3D volume from the individual 2D predictions saved as images.
To stitch the `.png` back to a nifti file:
```
$ python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
```

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](3dslicer_zoom.png)


### Plotting the metrics
There are some facilities to plot the metrics saved by [`main.py`](main.py):
```
$ python plot.py --help
usage: plot.py [-h] --metric_file METRIC_MODE.npy [--dest METRIC_MODE.png] [--headless]

Plot data over time

options:
  -h, --help            show this help message and exit
  --metric_file METRIC_MODE.npy
                        The metric file to plot.
  --dest METRIC_MODE.png
                        Optional: save the plot to a .png file
  --headless            Does not display the plot and save it directly (implies --dest to be provided.
$ python plot.py --metric_file results/segthor/ce/dice_val.npy --dest results/segthor/ce/dice_val.png
```
![Validation DSC](dice_val.png)

Plotting and visualization ressources:
* [Scientific visualization Python + Matplotlib](https://github.com/rougier/scientific-visualization-book)
* [Seaborn](https://seaborn.pydata.org/examples/index.html)
* [Plotly](https://github.com/plotly/plotly.py)

## Submission and scoring
Groups will have to submit:
* archive of the git repo with the whole project, which includes:
    * pre-processing;
    * training;
    * post-processing where applicable;
    * inference;
    * metrics computation;
    * script fixing the data using the matrix `AFF` from `affine.py` (or rather its inverse);
    * (bonus) any solution fixing patient27 without recourse to `affine.py`;
    * (bonus) any (even partial) solution fixing the whole dataset without recourse to `affine.py`;
* the best trained model;
* predictions on the [test set](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EWZH7ylUUFFCg3lEzzLzJqMBG7OrPw1K4M78wq9t5iBj_w?e=Yejv5d) (`sha256sum -c data/test.zip.sha256` as optional checksum);
* predictions on the group's internal validation set, the labels of their validation set, and the metrics they computed.

The main criteria for scoring will include:
* improvement of performances over baseline;
* code quality/clear [git use](git.md);
* the [choice of metrics](https://metrics-reloaded.dkfz.de/) (they need to be in 3D);
* correctness of the computed metrics (on the validation set);
* (part of the report) clear description of the method;
* (part of the report) clever use of visualization to report and interpret your results;
* report;
* presentation.

The `(bonus)` lines give extra points, that can ultimately compensate other parts of the project/quizz.


### Packing the code
`$ git bundle create group-XX.bundle master`

### Saving the best model
`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

### Archiving everything for submission
All files should be grouped in single folder with the following structure
```
group-XX/
    test/
        pred/
            Patient_41.nii.gz
            Patient_42.nii.gz
            ...
    val/
        pred/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        gt/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        metric01.npy
        metric02.npy
        ...
    group-XX.bundle
    bestmodel-group-XX.pkl
```
The metrics should be numpy `ndarray` with the shape `NxKxD`, with `N` the number of scan in the subset, `K` the number of classes (5, including background), and `D` the eventual dimensionality of the metric (can be simply 1).

The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:
```
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
$ tar cf group-XX.tar.gz - group-XX/
```


## Known issues
### Cannot pickle lambda in the dataloader
Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

### Pytorch not compiled for Numpy 2.0
It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

### Viewer on Windows
Windows has different paths names (`\` in stead of `/`), so the default regex in the viewer needs to be changed to `--id_regex=".*\\\\(.*).png"`.
