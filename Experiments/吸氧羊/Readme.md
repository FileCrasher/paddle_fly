# PaddleClas实现图像分类Baseline

学习尝试了图片分类的任务，尝试了食物识别的项目，数据集采用课程提供的食物5类.zip

# 一、项目背景

一方面是想要提升自己的学习能力，另一方面是想要尝试对食物识别从而可以应用到餐厅等方面，可在自动收费访民啊进行应用

# 二、数据集简介

使用的Aistudio官方的食物数据集，包含了五个大类，分别是
beef_carpaccio
baby_back_ribs
beef_tartare
apple_pie
baklava
共一千多张图片

## 1.数据加载和预处理


```python
import os
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# train_list.txt（训练集）
# val_list.txt（验证集）
# 先把路径搞定 比如：foods/beef_carpaccio/855780.jpg ,读取到并写入txt 

# 根据左侧生成的文件夹名字来写根目录
dirpath = "foods"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0 # 标记总文件数量
    j = 0 # 标记文件类别
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（数字类别）（注意：中间有空格）。              
            imgpath = os.path.join(root,file)
            all_list.append(imgpath+" "+str(j)+"\n")

        j = j + 1

    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i
all_list,all_lenth = get_all_txt()
print(all_lenth)
```



## 2.数据打乱


```python
# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")
```


## 3.数据划分

```python
train_size = int(all_lenth * 0.9)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))
```

# 三、模型选择和开发

使用的是ShuffleNetV2网络

## 1.模型组网

![](https://ai-studio-static-online.cdn.bcebos.com/08542974fd1447a4af612a67f93adaba515dcb6723ff4484b526ff7daa088915)


```python
# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)
```

## 2.模型网络结构可视化


```python
# 模型封装
model = paddle.Model(network)

# 模型可视化
model.summary((1, 28, 28))
```

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #    
    ===========================================================================
       Flatten-1       [[1, 28, 28]]           [1, 784]              0       
       Linear-1          [[1, 784]]            [1, 512]           401,920    
        ReLU-1           [[1, 512]]            [1, 512]              0       
       Linear-2          [[1, 512]]            [1, 10]             5,130     
    ===========================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    ---------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.01
    Params size (MB): 1.55
    Estimated Total Size (MB): 1.57
    ---------------------------------------------------------------------------
    
    {'total_params': 407050, 'trainable_params': 407050}

## 3.修改配置文件


```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用GPU训练
  device: gpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 20 
  print_batch_step: 1
  use_visualdl: True #开启可视化（目前平台不可用）
  # used for static mode and model export
  # 图像大小
  image_shape: [3, 224, 224] 
  save_inference_dir: ./inference
  # training model under @to_static
  to_static: False

# model architecture
Arch:
  # 采用的网络
  name: ResNet50
  # 类别数 多了个0类 0-5 0无用 
  class_num: 6 
 
# loss function config for traing/eval process
Loss:
  Train:

    - CELoss: 
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.015
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0005


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的训练集文本路径
      cls_label_path: ./dataset/foods/train_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的验证集文本路径
      cls_label_path: ./dataset/foods/val_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: ./dataset/foods/beef_carpaccio/855780.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    # 输出的可能性最高的前topk个
    topk: 5
    # 标签文件 需要自己新建文件
    class_id_map_file: ./dataset/label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.0325 - acc: 0.9902 - 7ms/step           
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 157/157 [==============================] - loss: 7.0694e-04 - acc: 0.9807 - 6ms/step     
    Eval samples: 10000



## 4.模型评估测试


```python
# 模型评估，根据prepare接口配置的loss和metric进行返回result = model.evaluate(eval_dataset, verbose=1)print(result)
```

    Eval begin...The loss value printed in the log is the current batch, and the metric is the average value of previous step.step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9795 - 2ms/step         Eval samples: 10000{'loss': [0.0], 'acc': 0.9795}


## 5.模型预测

### 5.1 批量预测

使用model.predict接口来完成对大量数据集的批量预测。


```python
# 进行预测操作
result = model.predict(eval_dataset)

# 定义画图方法
def show_img(img, predict):
    plt.figure()
    plt.title('predict: {}'.format(predict))
    plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
    plt.show()

# 抽样展示
indexs = [2, 15, 38, 211]

for idx in indexs:
    show_img(eval_dataset[idx][0], np.argmax(result[0][idx]))
```

    Predict begin...
    step 10000/10000 [==============================] - 1ms/step        
    Predict samples: 10000


### 5.2 单张图片预测

采用model.predict_batch来进行单张或少量多张图片的预测。


```python
# 读取单张图片
image = eval_dataset[501][0]

# 单张图片预测
result = model.predict_batch([image])

# 可视化结果
show_img(image, np.argmax(result))
```

# 四、效果展示

直接将图片放在测试文件夹下，然后运行检测程序就可以识别分类出不同食物的类别。

就目前来看，识别率较高，以下是识别结果

```
[{'class_ids': [5, 1, 3, 4, 2],
'scores': [0.48433, 0.26765, 0.13903, 0.05609, 0.05162],
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 
'label_names': ['baklava', 'beef_carpaccio', 'beef_tartare', 'apple_pie', 'baby_back_ribs']}]
```

# 五、总结与升华

在学习过程中遇到了很多困难，但还是大致学会了PaddlePaddle和Aistudio的使用，不禁让我感叹飞桨系开发平台的高效与强大。
期待日后能继续学习飞桨，用它做出更多有趣的作品。

# 个人简介

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ [https://aistudio.baidu.com/aistudio/personalcenter/thirdview/800410](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/800410)


```python

```
