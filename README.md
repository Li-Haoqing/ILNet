# ILNet: Low-level Matters for Salient Infrared Small Target Detection


## Requirements
     Python 3.7.10
     torch 1.10.1

## Training

``python train.py --img_size 512 --batch_size 8 --epochs 600 --warm_up_epochs 10 --learning_rate 0.001 --dataset sirst --mode 'L' --amp True``

In default, the _'.pth'_ will be saved at `` ./results_sirst/ `` or `` ./results_IRSTD-1k/ ``.


## Valuating

``python val.py --img_size 512 --dataset 'sirst' --batch-size 1 --mode 'L' --checkpoint ' .pth' ``


## Demo

``python demo.py --img_path ' .png' --mask_path ' .png' --mode 'L' --checkpoint ' .pth' ``

## Datasets
Dataset folder should be like:

https://github.com/RuiZhang97/ISNet
~~~
IRSTD-1k
└───imges
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───masks
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───trainval.txt
└───test.txt
~~~
https://github.com/YimianDai/sirst
~~~
SIRST
└───idx_320
│       │   trainval.txt
│       │   test.txt
└───idx_427
│       │   trainval.txt
│       │   test.txt
└───imges
│       │   Misc_1.png
│       │   Misc_2.png
│       │  ...
└───masks
│       │   Misc_1_pixels0.png
│       │   Misc_2_pixels0.png
│       │  ...
~~~


## Best Results
SIRST
Mode      | Best IoU(%) | Best nIoU(%)  | Best Pd(%)  | Best Fa(1e-6)
---       | ---         | ---           | ---         | ---
ILNet-S   | 78.12       |  76.42        |  99.07      |  5.50
ILNet-M   | 79.57       |  77.19        |  98.15      |  3.02
ILNet-L   | 80.31       |  78.22        |  100        |  1.33
---       | ---         | ---           | ---         | ---
IRSTD-1k
Mode      | Best IoU(%) | Best nIoU(%)  | Best Pd(%)  | Best Fa(1e-6)
---       | ---         | ---           | ---         | ---
ILNet-S   | 66.01       |  64.78        |  93.27      |  5.26
ILNet-M   | 67.86       |  68.40        |  94.61      |  5.09
ILNet-L   | 70.15       |  68.91        |  95.29      |  3.23


## Thanks:
     Part of the code draws on the work of the following authors:
        https://github.com/Tianfang-Zhang/acm-pytorch
        https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation/u2net
     
     Datasets:
        https://github.com/YimianDai/sirst
        https://github.com/RuiZhang97/ISNet

     Metrics:
        https://github.com/YimianDai/sirst
        https://github.com/Lliu666/DNANet_BatchFormer

## Citation:
     @ARTICLE{ilnet,
       author={Li, Haoqing and Yang, Jinfu and Wang, Runshi and Xu, Yifei},
       journal={IEEE Transactions on Aerospace and Electronic Systems}, 
       title={ILNet: Low-Level Matters for Salient Infrared Small Target Detection}, 
       year={2025},
       volume={},
       number={},
       pages={1-13},
       doi={10.1109/TAES.2025.3544613}}

     @article{li2023ilnet,
       title={ILNet: Low-level matters for salient infrared small target detection},
       author={Li, Haoqing and Yang, Jinfu and Wang, Runshi and Xu, Yifei},
       journal={arXiv preprint arXiv:2309.13646},
       year={2023}
     }

