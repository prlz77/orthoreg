# OrthoReg
LuaTorch implementation of orthoreg.

## Usage
1. Copy the contents of the [backward-hook](https://github.com/prlz77/orthoreg/blob/master/backward-hook.lua) in your code.
2. Add the beta (regularization strength, default 0.01), lambda (regularizer radius of influence, default 10), and epsilon (for numerically stable normalization), to the arguments of your script.
3. run

## Citation
```
@inproceedings{rodriguez2017regularizing,
  title={Regularizing CNNs with Locally Constrained Decorrelations},
  author={Rodr{\'{i}}guez, Pau and Gonz{\`{a}}lez, Jordi and Cucurull, Guillem and Gonfaus, Josep M. and Roca, Xavier},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

## Read more
* [Arxiv](https://arxiv.org/abs/1611.01967v2)
* [blog](https://prlz77.github.io/iclr2017-paper/)
