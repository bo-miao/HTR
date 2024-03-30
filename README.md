[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![arXiv](https://img.shields.io/badge/cs.CV-%09arXiv%3A2205.00823-red)](https://arxiv.org/abs/2403.19407)


The official implementation of the paper: 

<div align="center">
<h2>
<b>
Towards Temporally Consistent Referring Video Object Segmentation
</b>
</h2>
</div>

## Introduction

Referring Video Object Segmentation (R-VOS) methods face challenges in maintaining consistent object segmentation due to temporal context variability and the presence of other visually similar objects. 
We propose an end-to-end R-VOS paradigm that explicitly models temporal instance consistency alongside the referring segmentation. 
Furthermore, we propose a new Mask Consistency Score (MCS) metric to evaluate the temporal consistency of video segmentation. Extensive experiments demonstrate that our approach enhances temporal consistency by a significant margin, leading to top-ranked performance on popular R-VOS benchmarks.
## Installation and Data Preparation

Please refer to [SgMg](https://github.com/bo-miao/SgMg) for installation and data preparation.

## Evaluation

The checkpoint for HTR w/ SwinL is available at  [HTR-SwinL](https://drive.google.com/file/d/1dfUuU67NyHV302KfPYMw286nk5VebjG7/view?usp=sharing).

If you want to evaluate HTR on DAVIS, please run the following command in the `scripts` folder:

```
sh dist_test_davis_swinl.sh
```


## MCS Metric for Temporal Consistency

The code for MCS evaluation is in `get_mcs.py`. 
Please click `View scoring output log` to download `stdout.txt` of your submission in [Ref-YTVOS eval server](https://codalab.lisn.upsaclay.fr/competitions/3282).

Then you can run the script to get the MCS score under different thresholds.

## Citation

```
@article{miao2024htr,
  title={Towards Temporally Consistent Referring Video Object Segmentation},
  author={Miao, Bo and Bennamoun, Mohammed and Gao, Yongsheng and Shah, Mubarak and Mian, Ajmal},
  journal={https://arxiv.org/abs/2403.19407},
  year={2024}
}
```

## Acknowledgements

- [SgMg](https://github.com/bo-miao/SgMg)
- [ReferFormer](https://github.com/wjn922/ReferFormer)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)

## Contact
If you have any questions about this project, please feel free to contact bomiaobbb@gmail.com.


