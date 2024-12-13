# 模型训练

SSCMA 提供了多种算法，您可以根据自己的需求选择合适的算法，然后通过训练、导出和部署模型来解决实际问题。本章将进一步介绍如何使用 SSCMA 来训练、导出和部署模型。


## 参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/train.py --help

usage: train.py [-h] [--amp] [--auto-scale-lr] [--resume] [--work_dir WORK_DIR] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
                config

Train a detector

positional arguments:
  config                train config file path

options:
  -h, --help            show this help message and exit
  --amp                 enable automatic-mixed-precision training
  --auto-scale-lr       enable automatically scaling LR.
  --resume              resume from the latest checkpoint in the work_dir automatically
  --work_dir WORK_DIR, --work-dir WORK_DIR
                        the dir to save logs and models
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into
                        config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also
                        allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and
                        that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```
