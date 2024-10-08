# [TPAMI 2024] Learning with Constraint Learning: New Perspective, Solution Strategy and Various Applications [[Paper]](https://ieeexplore.ieee.org/abstract/document/10430445/)
By Risheng Liu<sup>1*,2</sup>, Jiaxin Gao<sup>1</sup>, Xuan Liu<sup>1</sup>, Xin Fan<sup>1</sup></h4> 

<sup>1</sup>Dalian University of Technology, <sup>2</sup>Peng Cheng Laboratory


## Pipeline
![](./Figures/pipeline.png)


### Dependencies
You can simply run the following command automatically install the dependencies

```pip install -r requirement.txt ```

This code mainly requires the following:
- Python 3.*
- tqdm
- Pytorch
- [higher](https://github.com/facebookresearch/higher) 

 
 
### Usage

You can run the python file for different applications following the script below:
1. GAN and Its Variants
```
Python unrolled_gan_rhg_ring_LwCL.py  # For 2D Ring MOG dataset.
Python  unrolled_gan_rhg_cube_LwCL.py  # For 3D Cube MOG dataset.
```

2. Multi-Task Meta-Learning Few-Shot Classification

For the few-shot classification experiments in multi-task meta-learning, the entire network architecture is based on the [L2F](https://github.com/baiksung/L2F) network. You can download the complete code from [Baidu Yun (extraction code: i06p)](https://pan.baidu.com/s/1ACtg6W1nnEsZI9LDB9lPTQ). The datasets used are  [mini-Imagenet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) and  [Omniglot](https://github.com/brendenlake/omniglot). 
Please download the corresponding dataset, for example, **mini_imagenet_full_size.tar.bz2**, and place it in the **dataset** directory. 
Then, execute the following command:

```
python train_maml_system.py --name_of_args_json_file experiment_config/mini-imagenet_l2f_mini-imagenet_5_way_1_shot_0_resnet12_GN.json --gpu_to_use 0  # For few-shot classification tasks.
```

3. Hyper-Parameter Learning

For hyper-cleaning experiments, download the corresponding datasets, and place it in the **dataset** directory. 
Then, execute the following command:

```
python ./HPL/data_hyper_cleaning.py  # For data hyper-cleaning tasks.
```

## Partial Results
1. Numerical mechanism evaluation:
<p align="center">
    <img src="./Figures/4-1.png" alt="Figure 4-1" width="35%" style="margin-right: 10px;"/>
    <img src="./Figures/4-2.png" alt="Figure 4-2" width="35%"/>
</p>
2. GAN and Its Variants: 
<p align="center">
    <img src="./Figures/MoG.png" alt="CIFAR comparison" width="70%">
</p>

## Citation
If you feel this project is helpful, please consider cite our paper :blush:
```
@article{liu2024learning,
  title={Learning with Constraint Learning: New Perspective, Solution Strategy and Various Applications},
  author={Liu, Risheng and Gao, Jiaxin and Liu, Xuan and Fan, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={7},
  pages={5026--5043},
  year={2024},
  publisher={IEEE}

}

@article{liu2021investigating,
  title={Investigating bi-level optimization for learning and vision from a unified perspective: A survey and beyond},
  author={Liu, Risheng and Gao, Jiaxin and Zhang, Jin and Meng, Deyu and Lin, Zhouchen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={12},
  pages={10045--10067},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [IAPTT-GM](https://github.com/vis-opt-group/IAPTT-GM), [L2F](https://github.com/baiksung/L2F) and [BLO](https://github.com/vis-opt-group/BLO). We thank the authors for sharing the codes for their great works.

If you have any inquiries, feel free to reach out to Jiaxin Gao via email at jiaxinn.gao@outlook.com, or contact Xuan Liu at liuxuan_16@126.com.

