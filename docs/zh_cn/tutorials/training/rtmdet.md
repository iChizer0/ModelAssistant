# RTMDet 模型训练

RTMDet (Real-time Models for Object Detection) 是一个高精度、低延时的单阶段目标检测器算法，RTMDet 模型整体结构和 YOLOX 几乎一致，由 CSPNeXt + CSPNeXtPAFPN + 共享卷积权重但分别计算 BN 的 SepBNHead 构成。内部核心模块也是 CSPLayer，但对其中的 Basic Block 改进为了 CSPNeXt Block。

## 数据集准备

在进行 RTMDet 模型训练之前，我们需要准备好数据集。这里我们以已经标注好的口罩 COCO 数据集为例，您可以在 [SSCMA - 公共数据集](../../datasets/public#获取公共数据集) 中下载该数据集。

## 模型选择与训练

SSCMA 提供了多种不同的 RTMDet 模型配置，您可以根据自己的需求选择合适的模型进行训练。

```sh
rtmdet_l_8xb32_300e_coco.py
rtmdet_m_8xb32_300e_coco.py
rtmdet_mnv4_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco_relu.py
rtmdet_nano_8xb32_300e_coco_relu_q.py
rtmdet_s_8xb32_300e_coco.py
```

在此我们以 `rtmdet_nano_8xb32_300e_coco.py` 为例，展示如何使用 SSCMA 进行 RTMDet 模型训练。

```sh
python tools/train.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    max_epochs=100
```

- `configs/rtmdet_nano_8xb32_300e_coco.py`: 指定配置文件，定义模型和训练设置。
- `--cfg-options`: 用于指定额外的配置选项。
    - `data_root`: 设定数据集的根目录。
    - `num_classes`: 指定模型需要识别的类别数量。
    - `train_ann_file`: 指定训练数据的注释文件路径。
    - `val_ann_file`: 指定验证数据的注释文件路径。
    - `train_img_prefix`: 指定训练图像的前缀路径。
    - `val_img_prefix`: 指定验证图像的前缀路径。
    - `max_epochs`: 设置训练的最大周期数。

等待训练结束后，您可能看到以下日志:

:::details

```sh
12/13 08:03:57 - mmengine - INFO - Epoch(train) [100][60/60]  base_lr: 3.2500e-05 lr: 3.2500e-05  eta: 0:00:00  time: 0.0855  data_time: 0.0015  memory: 625  loss: 0.7791  loss_cls: 0.4587  loss_bbox: 0.3204
12/13 08:03:57 - mmengine - INFO - Saving checkpoint at 100 epochs
12/13 08:03:59 - mmengine - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.32s).
Accumulating evaluation results...
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.910
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.624
12/13 08:03:59 - mmengine - INFO - bbox_mAP_copypaste: 0.515 0.910 0.530 -1.000 0.002 0.525
12/13 08:03:59 - mmengine - INFO - Epoch(val) [100][6/6]    coco/bbox_mAP: 0.5150  coco/bbox_mAP_50: 0.9100  coco/bbox_mAP_75: 0.5300  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: 0.0020  coco/bbox_mAP_l: 0.5250  data_time: 0.0257  time: 0.1672
```

:::

在 `work_dirs/rtmdet_nano_8xb32_300e_coco` 目录下找到训练好的模型。


## 模型导出及验证

在训练过程中，您可以随时查看训练日志、导出模型并验证模型的性能，部分模型验证中输出的指标在训练过程中也会显示，因此在这一部分我们会先介绍如何导出模型，然后阐述如何验证导出后模型的精度。

### 导出模型

这里我们以导出 TFLite 模型为例，您可以使用以下命令导出不同精度的 TFLite 模型：

```sh
python3 tools/export.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    --imgsz 192 192 \
    --format tflite
```

### 验证模型

导出完成后，您可以使用以下命令对模型进行验证：

```sh
python3 tools/test.py configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/
```

得到以下输出:

```sh
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.910
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.624
```

:::tip

关于以上输出的详细解释，请参考 [COCO 数据集评估指标](https://cocodataset.org/#detection-eval)，在这里我们主要关注 50-95 IoU 和 50 IoU 的 mAP。

:::



