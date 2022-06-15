# OMG: 3D Scene Geometry Decomposition and Manipulation from 2D Images


This repository contains the implementation of the paper:

**OMG: 3D Scene Geometry Decomposition and Manipulation from 2D Images** <br />
[Bing Wang](https://www.cs.ox.ac.uk/people/bing.wang/), [Lu Chen](https://chenlu-china.github.io/), [Bo Yang<sup>*</sup>](https://yang7879.github.io/) <br />
[**Paper**]() | [**Supplementary**]() | [**Video**]()

**Demo:** <br />

<!-- | ![2](./figs/decompose.gif)   | ![z](./figs/decompose.gif) | -->

<div align=center>
     <img src="/figs/decomposition.gif" width=70% > <br/></br>
     <strong><center>Reconstruction and Instances Decomposition</center></strong>     <br/><br/>

</div>

<div align=center>
     <img src="/figs/manipulation.gif" width=70%>  <br/></br>
     <strong><center>Objects Manipulation</center></strong>     <br/><br/>
</div>



## Installation

Create python environment vision >=3.7
```bash
pip install pytorch==1.8.1 torchVision==0.9.1 torchaudio===0.8.1
pip install -r environment.txt
```

## Datasets

To evaluate our model or train a new model from scratch, you have to obtain the respective dataset.
In this paper, we consider 3 different datasets:

### Scannet

If you need all the pre-processed files of ScanNet, you can access licenses of [ScanNet](https://github.com/ScanNet/ScanNet).

We used `scene0010_00, scene0012_00, scene0024_00, scene0033_00, scene0038_00, scene0088_00, scene0113_00, scene0192_00`.


### Replica

We used `office0, office2, office3, office4, room0, room1, room2`.


### OMG-SR（Ours）

Our dataset rendered by Blender(v2.82.7). It contains 8 normal kinds of indoor scenes, which are `Bathroom, Bedroom, Dinning, Kitchen, Reception, Rest, Study, Office`.

### Hints

Baisc information of each scenes can renference `./configs/0000/xxxx/xxxx.txt`.

## Training

You can change the your path and parameters under `./configs/xxxx/dataset_name/scene_name.txt`.

After you set all parameters you want, you can train model use one of blow command, for example:

For scannet:

If you want use full of segementation function, you can run commands like:
```bash

CUDA_VISIBLE_DEVICES=0 python train_scans_penalize.py --config configs/0050/scene0010_00.txt
or use nohup:
CUDA_VISIBLE_DEVICES=0 nohup python -u train_scans_penalize.py --config configs/0050/scene0010_00.txt > logs_0050/scene0010_00.txt 2>&1 &

```
If you do not segement non-occupied area, you can run commands like:

```bash

CUDA_VISIBLE_DEVICES=0 python train_replica.py --config configs/0050/scene0010_00.txt
or use nohup:
CUDA_VISIBLE_DEVICES=0 nohup python -u train_replica.py --config configs/0050/scene0010_00.txt > logs_0050/scene0010_00.txt 2>&1 &

```

## Testing and Editor Testing

For 2D testing, we used PSNR, SSIM, LPIPS, and mAP to evaluate our task:

You need to add `render=True` and `log_time="your log folder name"` into config txt, and then run `CUDA_VISIBLE_DEVICES=0 python test_xxxx.py --config configs/0050/scene_name.txt`.

For editor testing:

Change `render=True` to 'edito_render=True', and eidt a object_info.json to assign objects you want to edior, specific format can renference `./editor_configs/omg-sr/study_room.txt` and `./data/omg-sr/study_room/object_info.json`.

run `CUDA_VISIBLE_DEVICES=0 python editor_test_xxxx.py --config configs/0050/scene_name.txt`.

## Baseline

SOTA method Mask R-CNN

## Qualitative Results

<div align=center>
     <img src="/figs/results.png" width=100% > <br/></br>
</div>

## Camera and Object Rotating

<div align=center>
     <img src="/figs/study_room1.gif" width=30%> <br/></br>
</div>

## Citation
If you find our work useful in your research, please consider citing:

