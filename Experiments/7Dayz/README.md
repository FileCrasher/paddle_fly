# 项目简介
## 1背景

<img src='https://ai-studio-static-online.cdn.bcebos.com/e39d60087f964cfe9e2aceef1b158a7c773835a46c5545d7a7ead927738b8dca'/>


>为了防止类似的塌陷再次发生，为了守护国民的安全，拟定这里有一款无人机，它能够搭载GPS模块，搭载算力机构(如Nano)，一边飞一边检测桥面或路面的裂痕缺陷，一旦检测到了就发送实时位置给工程师，让抢修队及时到达现场...那将多么梦幻！

>本应该是全套流程下来加上部署，但因训练时间赶不上DDL,所以只取其中的bridge_crack_1中的600多张图片进行训练，而且没有硬件支持，部署大概也就做个模型转换


## 2参考项目
- [PaddleSeg：华录杯·车道线检测（单卡、多卡训练）](https://aistudio.baidu.com/aistudio/projectdetail/1081298?channelType=0&channel=0)

# 数据集介绍
### 1.1缺陷样例
<img src='https://ai-studio-static-online.cdn.bcebos.com/13496de92a8c404687c5284adaccdead89ac66b0fabd45ab821530ed4aca925d'/>

### 1.2标注示例
采用EasyDL标注,过程中标注质量参差不齐,怀疑团队里有🐖

建议提前协商好统一的标注工具

| 标注前 | 标注后 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/af437a0dc1f94b62b3fec2f67ab91d1a19d889f34fea44028c53605f1820a8ae) | ![](https://ai-studio-static-online.cdn.bcebos.com/01a344215127408fa1a5b0b94439f42e8c5049bba8f94b65833edaa8a0abb707) |


### 1.3评审规则
$$mIoU = \frac{1}{C}\sum_{c=1}^C{IoU_c}\tag{1.1}$$
$$IoU_c = \frac{TP}{TP+FP+FN} \tag{1.2}$$
$$TP = {\sum_i}{\parallel{M_ci\cdot{M_ci^*}}\parallel}_0\tag{1.3}$$
$$FP = {\sum_i}{\parallel{M_ci\cdot{(1-M_ci^*)}}\parallel}_0\tag{1.4}$$
$$TP = {\sum_i}{\parallel{(1-M_ci)\cdot{M_ci^*}}\parallel}_0\tag{1.5}$$
$其中，C是分类数，在本任务中就是缺陷的类别数，TP, FP 和TN 表示true-positive, false-positive 和 false-negative。$

## 1.环境配置

- PaddleSeg release 2.2
- PaddlePaddle 2.0.2


```python
!git clone https://gitee.com/paddlepaddle/PaddleSeg.git
#此操作会刷新你的PaddleSeg配置
```


```python
%cd PaddleSeg
!pip install -r requirements.txt
```

## 2.解压数据集
真的很解压

### 2.1数据转换
<font color=red size=5>注：标注后的数据集往往只生成一个.json文件和原图.png，如果使用LabelMe标注，请使用PaddleSeg->tools->labelme2seg.py转换数据为PaddleSeg可用的格式。如果用EasyDL的多边形工具，则用我改动的PaddleSeg->tools->easydl2seg.py</font>
```
data                 # Root directory
|-- annotations            # Ground-truth
|   |-- xxx.png            # Pixel-level truth information
|   |...
|-- class_names.txt        # The category name of the dataset
|-- xxx.jpg(png or other)  # Original image of dataset
|-- ...
|-- xxx.json               # Json file,used to save annotation information
|-- ...
```
名字"annotations", "class_names.txt"不要轻易改动，否则更改源码


```python
# 解压数据集
!unzip ../data/data104136/bridge_crack_1.zip -d ./dataset/bridge
```


```python
#解压数据集
!unzip ../data/data104352/data.zip -d ./dataset/bridge
```


```python
#此行转换数据集，生成mask在annotations里
!python PaddleSeg/tools/easydl2seg.py PaddleSeg/dataset/bridge/data
```

### 2.2看一眼生成的mask
> 颜色有点阴间，你可以自己在标注的时候设置颜色

<img src="https://ai-studio-static-online.cdn.bcebos.com/a950ce96038943ef8c032ec739c1dd917cb5516f51324781b49c5b7cf85a089e" width='300' height="300"/>

## 3.转换&划分数据集
### 3.1转换数据集为PaddleSeg可用的形式。转换时遇到大坑：
- EasyDL标注后的原始数据不能直接用于PaddleSeg下```tools/labelme2seg.py```，需要自己更改源码```PaddleSeg/tools/easydl2seg.py```。目前只支持多边形转换，brush工具不会。
- 标注一定要多找些人
- 使用```python tools/labelme2seg.py legacy/docs/annotation/labelme_demo/```时后面的绝对路径不能带中文，不然转换的时候会报错


```python
!python makelist.py
#此时生成了train_list.txt(80%) val_list.txt(20%)
```

    2280108.ipynb  data  PaddleSeg	work


## 4.开始训练
### 4.1选择模型
选择Cityscapes预训练模型
|模型|数据集合|下载地址|outer stride|
|---|---|---|---|
|DeepLabv3+/MobileNetv3_Large/bn|Cityscapes|[deeplabv3p_mobilenetv3_large_cityscapes.tar.gz](https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz)|32|
#### DeepLabv3+模型介绍
>DeepLabv3+ [2] 是DeepLab系列的最后一篇文章，其前作有 DeepLabv1, DeepLabv2, DeepLabv3. 在最新作中，作者通过encoder-decoder进行多尺度信息的融合，以优化分割效果，尤其是目标边缘的效果。 并且其使用了Xception模型作为骨干网络，并将深度可分离卷积(depthwise separable convolution)应用到atrous spatial pyramid pooling(ASPP)中和decoder模块，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 和 Cityscapes 数据集上取得新的state-of-art performance.

![](https://ai-studio-static-online.cdn.bcebos.com/9fac26f996194e93adef88f60ee17088fadc02aaad294e4880af0c0e3e245112)

### 4.2[类别不均衡问题](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.2/docs/lovasz_loss.md)处理：
> 在图像分割任务中，经常出现类别分布不均匀的情况

> 例如：工业产品的瑕疵检测、道路提取及病变区域提取等。

> 我们可使用lovasz loss解决这个问题。[参考文献](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html)

>Lovasz loss根据分割目标的类别数量可分为两种：```lovasz hinge loss```和```lovasz softmax loss```. 其中```lovasz hinge loss```适用于二分类问题，```lovasz softmax loss```适用于多分类问题。我们本次只有两类：\_background_和crack类，选用```lovasz hinge loss```

- 参考yml文件，输入图像大小为1024x1024，故选用相应cityscapes预训练模型：

- 时间足够的话，设置iters为40000或许能再往后磨精度
```_base_: '../_base_/cityscapes_1024x1024.yml'

batch_size: 2
iters: 10000

model:
  type: DeepLabV3P
  backbone:
    type: ResNet101_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained:  https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz
  num_classes: 2
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]
  aspp_out_channels: 256
  align_corners: True
  pretrained: null

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5



learning_rate:
  value: 0.001
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0001


loss:
  types:
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.8, 0.2]
  coef: [1]


train_dataset:
  type: Dataset
  dataset_root: /home/aistudio/
  train_path: train_list.txt
  num_classes: 2
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [1024, 1024]
    - type: RandomHorizontalFlip
    - type: Normalize
  mode: train

val_dataset:
  type: Dataset
  dataset_root: /home/aistudio/
  val_path: val_list.txt
  num_classes: 2
  transforms:
    - type: Normalize
  mode: val
```


```python
# 回到原始目录
%cd ~
```


```python
!python PaddleSeg/train.py \
--config PaddleSeg/configs/deeplabv3p/deeplabv3p_MobileNetv2_512x512.yml \
--do_eval \
--use_vdl \
# --resume_model output/iter_137000/ 恢复训练用
```

| mIoU   | Acc    | Kappa  |
| ------ | ------ | ------ |
| 0.5013 | 0.9595 | 0.0760 |

⭐性能指标如上


## 5.可视化训练（下次一定

## 6.Pridict With TTA
>TTA-Test Time Augmentation(测试时增强)
> 和数据增广相似，TTA给你的测试数据进行了增广。目的是给你的测试数据随机调整(如下面的flip_horizontal，flip_vertical)。因此，除了常规的，“干净的”测试图像外，我们还会喂给训练后的模型增广后的图像，然后把一张图像的所有预测结果都列出来，取我们分数最高的作为最终结果，也算是提高ACC一个取巧的方法

这个图非常形象👇：FROM--[GT](https://github.com/AgentMaker/PaTTA)

```    
	   Input
             |           # input batch of images 
        / / /|\ \ \      # apply augmentations (flips, rotation, scale, etc.)
       | | | | | | |     # pass augmented batches through model
       | | | | | | |     # reverse transformations for each batch of masks/labels
        \ \ \|/ / /      # merge predictions (mean, max, gmean, etc.)
             |           # output batch of masks/labels
           Output
```


```python
# 预测代码
!python PaddleSeg/predict.py --config PaddleSeg/configs/deeplabv3p/deeplabv3p_MobileNetv2_512x512.yml \
--model_path output/best_model/model.pdparams \
--image_path infer \
--save_dir output/result\
--aug_pred \
--flip_horizontal \
--flip_vertical
```


可视化预测结果：一言难尽。果然用600多张训练还是太少了

改进：

- 训练数据预计增加到1800张
- 调参

<img src = "https://ai-studio-static-online.cdn.bcebos.com/520129bb0458436eae45fcf0ecc41b7e0dbd4910ee0d40e3a10c58c6ee40405e" width = "300" height = "300" />

## 7.导出模型
- 为后面OPT转换做准备

### 7.1模型导出为静态图
```
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```


```python
!python PaddleSeg/export.py \
       --config PaddleSeg/configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --model_path output/best_model/model.pdparams
```




## 8.部署在安卓端 & OPT工具转换PaddleLite模型
使用OPT工具转换参数为PaddleLite能用的格式


```python
!chmod +x ./opt_linux
!./opt_linux --model_file=output/model.pdmodel --param_file=output/model.pdiparams --optimize_out=Deeplab3p_MobileNetv3
```

    ......
    [I  8/15 23: 7:24.934 ...e-Lite/lite/model_parser/model_parser.cc:481 SaveModelNaive] Save naive buffer model in Deeplab3p_MobileNetv3.nb successfully


生成.nb文件

## 9🕳坑先埋在这里
部署等一波硬件和finetune

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
