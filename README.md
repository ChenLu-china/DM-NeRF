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
     <strong><center>Reconstruction and Instances Decomposition</center></strong>     <br/><br/>
</div>



## Installation

create python environment vision >=3.7
```bash
pip install pytorch==1.8.1 torchVision==0.9.1 torchaudio===0.8.1
pip install -r environment.txt
```

## Datasets

To evaluate our model or train a new model from scratch, you have to obtain the respective dataset.
In this paper, we consider 3 different datasets:

### Scannet

If you need all the pre-processed files of ScanNet, you can access licenses of [ScanNet](https://github.com/ScanNet/ScanNet).

we used scene0010_00, scene0012_00, scene0024_00, scene0033_00, scene0038_00, scene0088_00, scene0113_00, scene0192_00.


### Replica


we used office0, office2, office3, office4, room0, room1, room2


### Synthetic Indoor Rooms（Ours）

## Qualitative Results

## Camera and Object Rotating

<div align=center>
     <img src="/figs/study_room1.gif" width=30%> <br/></br>
</div>

## Citation
If you find our work useful in your research, please consider citing:

