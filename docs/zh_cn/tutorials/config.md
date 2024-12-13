# 模型配置

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 使用 [MMEngine](https://github.com/open-mmlab/mmengine) 提供的配置处理系统，具有模块化、可继承的设计特点，为用户提供了统一的配置访问接口，便于用户对不同的神经网络进行各种测试与验证。


## 配置的目录结构

SSCMA 使用的配置文件位于 `configs` 目录下，用于不同任务下不同模型的训练。我们在其根据不同的任务分类划分了子文件夹，在各个子文件夹中，保存有多个模型的不同训练管线参数，配置文件的目录结构如下:

:::code-group

```sh [整体结构]
├── anomaly
│   └── vae_mirophone.py
├── _base_
│   ├── default_runtime.py
│   └── schedules
│       ├── AdamW_linear_coslr_bs2048.py
│       ├── schedule_1x.py
│       └── sgd_linear_coslr_bs2048.py
├── datasets
│   ├── coco_detection.py
│   ├── imagenet_bs32.py
│   └── lancedb_bs32.py
├── fomo
│   ├── fomo_mobnetv2_0.35_abl_coco.py
│   └── fomo_mobnetv2_1_x16_coco.py
├── __init__.py
├── mobilenetv4_imagenet.py
├── models
│   └── timm_classify.py
├── pfld
│   ├── pfld_mbv2_1000e.py
│   └── pfld_mbv3l_192_1000e.py
├── pretrain
│   ├── default_runtime.py
│   ├── imagenet_bs2048_rsb.py
│   └── rtmdet_nano_8xb256_600e_coco_1k.py
├── resnet50_imagenet.py
├── rtmdet_l_8xb32_300e_coco.py
├── rtmdet_m_8xb32_300e_coco.py
├── rtmdet_mnv4_8xb32_300e_coco.py
├── rtmdet_nano_8xb32_300e_coco.py
├── rtmdet_nano_8xb32_300e_coco_relu.py
├── rtmdet_nano_8xb32_300e_coco_relu_q.py
└── rtmdet_s_8xb32_300e_coco.py
```

```sh [按不同算法分类] {1-2,13-15,17,19,20-22,25,26-34}
├── anomaly
│   └── vae_mirophone.py
├── _base_
│   ├── default_runtime.py
│   └── schedules
│       ├── AdamW_linear_coslr_bs2048.py
│       ├── schedule_1x.py
│       └── sgd_linear_coslr_bs2048.py
├── datasets
│   ├── coco_detection.py
│   ├── imagenet_bs32.py
│   └── lancedb_bs32.py
├── fomo
│   ├── fomo_mobnetv2_0.35_abl_coco.py
│   └── fomo_mobnetv2_1_x16_coco.py
├── __init__.py
├── mobilenetv4_imagenet.py
├── models
│   └── timm_classify.py
├── pfld
│   ├── pfld_mbv2_1000e.py
│   └── pfld_mbv3l_192_1000e.py
├── pretrain
│   ├── default_runtime.py
│   ├── imagenet_bs2048_rsb.py
│   └── rtmdet_nano_8xb256_600e_coco_1k.py
├── resnet50_imagenet.py
├── rtmdet_l_8xb32_300e_coco.py
├── rtmdet_m_8xb32_300e_coco.py
├── rtmdet_mnv4_8xb32_300e_coco.py
├── rtmdet_nano_8xb32_300e_coco.py
├── rtmdet_nano_8xb32_300e_coco_relu.py
├── rtmdet_nano_8xb32_300e_coco_relu_q.py
└── rtmdet_s_8xb32_300e_coco.py
```

:::

:::tip

其中名为 `_base_` 的任务文件夹是我们其他任务的继承对象，关于配置文件继承的详细说明，请参考 [MMEngine - 配置文件的继承](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#id3)。

:::


## 配置的内容结构

以 [FOMO 模型训练](./training/fomo)中的 `fomo_mobnetv2_0.35_x8_abl_coco.py` 配置文件为例，我们根据不同的功能模块介绍该配置文件中的各个字段。

它使用了Python的字典和列表来定义模型、数据加载、训练和评估等各个部分的参数。以下是该配置文件的结构和各个部分的作用，以及一般需要调整的参数：

- **导入模块**：
   - 导入了必要的模块和函数，如`read_base`、`CustomFomoCocoDataset`、`FomoInfer`等。

- **模型配置**：
   - `num_classes`：类别数量，对于口罩检测，通常是 2（戴口罩和不戴口罩）。
   - `widen_factor`：模型宽度因子，用于调整模型的宽度。

- **数据配置**：
   - `dataset_type`：指定数据集类型。
   - `data_root`：数据集的根目录。
   - `train_ann`、`train_data`、`val_ann`、`val_data`：训练和验证数据的注释文件和数据目录。
   - `height`、`width`、`imgsz`：输入图像的尺寸。

- **训练配置**：
   - `batch`、`workers`、`persistent_workers`：训练时的批处理大小、工作线程数和持久工作线程。
   - `val_batch`、`val_workers`：验证时的批处理大小和工作线程数。
   - `lr`、`epochs`、`weight_decay`、`momentum`：学习率、训练周期、权重衰减和动量。

- **钩子**：
   - `default_hooks`：定义了训练过程中的钩子，如可视化钩子。
   - `visualizer`：定义了可视化器。

- **数据预处理**：
   - `data_preprocessor`：定义了数据预处理的参数，如均值、标准差、颜色转换等。

- **模型结构**：
   - 定义了模型的类型、数据预处理器、骨干网络和头部网络的配置。

- **部署配置**：
   - `deploy`：定义了模型部署时的数据预处理器配置。

- **图像解码后端**：
   - `imdecode_backend`：指定图像解码的后端。

- **预处理流水线**：
    - `pre_transform`、`train_pipeline`、`test_pipeline`：定义了训练和测试数据的预处理流水线。

- **数据加载器**：
    - `train_dataloader`、`val_dataloader`、`test_dataloader`：定义了训练、验证和测试数据加载器的配置。

- **优化器配置**：
    - `optim_wrapper`：定义了优化器的类型和参数。

- **评估器**：
    - `val_evaluator`、`test_evaluator`：定义了验证和测试的评估器。

- **训练配置**：
    - `train_cfg`：定义了训练的配置，如是否按周期训练和最大周期数。

- **学习策略**：
    - `param_scheduler`：定义了学习率调度器的策略。

该配置文件涵盖了从数据预处理到模型训练和评估的各个方面。根据具体的训练需求，可能需要调整的参数包括学习率、批次大小、训练周期、优化器参数、数据增强策略等。这些参数的调整将直接影响模型的性能和训练效果。

### 重要参数

更改训练配置时，通常需要修改以下参数。例如，`height` 和 `width` 参数通常用于确定输入图像大小，其应与模型接受的输入尺寸保持一致，因此我们建议在配置文件中单独定义这些参数。

```python
height=96       # 输入图像高度
width=96        # 输入图像宽度
batch_size=8    # 验证期间单个 GPU 的批量大小
workers=2       # 验证期间单个 GPU 预读取数据的线程数
epoches=100     # 最大训练轮次: 100 轮
lr=0.02         # 学习率
```

### 网络结构

在 FOMO 模型的配置文件中，我们使用以下结构化的配置文件来设置检测算法组件，包括 Backbone、Neck 等重要的神经网络组件。部分模型配置如下:

```python
model = dict(
    type=Fomo,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MobileNetv2, widen_factor=widen_factor, out_indices=(2,), rep=False
    ),
    head=dict(
        type=FomoHead,
        input_channels=[16],
        num_classes=num_classes,
        middle_channel=48,
        act_cfg=ReLU,
        loss_cls=dict(type=BCEWithLogitsLoss, reduction="none"),
        loss_bg=dict(type=BCEWithLogitsLoss, reduction="none"),
    ),
    skip_preprocessor=True,
)
```

### 数据集和验证

在设置好网络模型后，我们还需要设置数据集和数据加载管道来构建数据加载器。由于这部分的复杂性，我们使用中间变量来简化数据加载器配置的编写。完整的数据增强方法可以在 `sscma/datasets/pipelines` 文件夹中找到。

我们将在这里演示 FOMO 的训练和测试管线，该管线使用了[自定义的 COCO_MASK 数据集](./datasets):

```python
dataset_type = CustomFomoCocoDataset
#...
pre_transform = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend=imdecode_backend,
        file_client_args=dict(backend="disk"),
    ),
    dict(type=LoadAnnotations, with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(type=HSVRandomAug),
    dict(
        type=Mosaic,
        img_scale=imgsz,
        use_cached=True,
        random_pop=False,
        pad_val=114.0,
    ),
    dict(type=Resize, keep_ratio=True, scale=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=Bbox2FomoMask, downsample_factor=downsample_factor, num_classes=num_classes
    ),
    dict(
        type=PackDetInputs,
        meta_keys=(
            "fomo_mask",
            "img_path",
            "img_id",
            "instances",
            "img_shape",
            "ori_shape",
            "gt_bboxes",
            "gt_bboxes_labels",
        ),
    ),
]
```

此外，我们还需要设置一个评估器。评估器用于计算训练模型在验证和测试数据集上的精度指标，其的配置由一个或一系列指标配置组成:

```python
val_evaluator = dict(type=FomoMetric)
test_evaluator = val_evaluator
```

### 优化器

```python
optim_wrapper = dict(
    optimizer=dict(
        type=SGD,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    ),
)
```

:::tip

关于 Hook 的更多应用细节，请参考 [MMEngine - Hook](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)。

:::

### 配置文件继承

在目录 `config/_base_` 下包含默认的配置文件，由 `_base_` 中的组件组成的配置文件称为原始配置。

为了便于测试，我们建议使用者继承现有的配置文件。例如，在 FOMO 模型的训练配置文件中设置有 `read_base()`，然后在继承文件的基础上，我们修改配置文件中的必要字段。

```python
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *
```


## 临时配置覆写在

使用 `tools/train.py` 或 `tools/test.py` 时，可以指定 `--cfg-options` 临时覆写配置。

:::tip

可以按照原始配置中字典键的顺序指定配置选项并更新字典链的配置键。例如 `--cfg-options data_root='./dataset/coco'` 更改数据集的数据根目录。

:::


## FAQs

- 不同模型的配置文件会有一定的差异,我如何理解?

  更多细节请参考 [MMDet Config](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html)，[MMPose Config](https://mmpose.readthedocs.io/zh_CN/latest/tutorials/0_config.html) 和 [MMCls Config](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。
