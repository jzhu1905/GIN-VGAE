# GIN-VGAE
## Usage
To obtain the test accuracies reported in the final project paper
```
# GIN-VGAE
python main.py --epochs 60 --dataset PTC
python main.py --epochs 50

# GIN
python main.py --epochs 60 --dataset PTC --model GIN
python main.py --epochs 50 --model GIN
```

## Reference
#### 1). GIN
```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```
Model implementation in this repo is adopted from [Pytorch Implementation by @weihua916
](https://github.com/weihua916/powerful-gnns)

#### 2). VGAE
```
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```
Model implementation in this repo is adopted from [Pytorch Implementation by @zfjsail](https://github.com/zfjsail/gae-pytorch/)