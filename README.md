# Multi-task-Depth-Estimation
This is the PyTorch implementation for our paper.

## Environment
1. Python 3.7.10
2. PyTorch 1.8.1
3. CUDA 11.1
4. Ubuntu 16.04

## Datasets
[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)


Prepare the two datasets according to the datalists (*.txt in [datasets](https://github.com/fyhfly/Multi-task-Depth-Estimation/tree/main/datasets))
```
datasets
  |----kitti 
         |----2011_09_26         
         |----2011_09_28        
         |----.........        
  |----kitti_predSemantics
         |----2011_09_26         
         |----2011_09_28        
         |----.........
```

## Training (RTX 3090, 24GB)

```
python train.py --model lv --gpu_ids 0 --niter 5 --niter_decay 45 --batchSize 8
```

## Test

```
python test.py --model lv --which_epoch 25 --gpu_ids 1
```

## Acknowledgments
Code is inspired by [GASDA](https://github.com/sshan-zhao/GASDA) and [MMoE](https://github.com/drawbridge/keras-mmoe).

## Contact
Qingshun Fu: qingshun.fu@bupt.edu.cn
