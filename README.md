# RaFM: Rank-Aware Factorization Machines

This is an implementation for the following paper:

Xiaoshuang Chen, Yin Zheng, Jiaxing Wang, Wenye Ma, Junzhou Huang. RaFM: Rank-Aware Factorization Machines. ICML 2019.

**Please cite our ICML 2019 paper if you find the paper and the codes beneficial for your research. Thanks!**

## Dependency
The code is based on Tensorflow, and we have tested it on Python 3.6, Tensorflow 1.13.

## Usage
Use statistics.py to compute the frequencies of the features occuring in the datasets.

Use RaFM.py to run the RaFM model.

Default parameters are provided, so you can execute them directly. You can also set it by yourself (see parse_args in each file).

