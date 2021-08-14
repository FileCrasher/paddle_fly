# ****PaddleX实现目标检测baseline

## 一、项目背景
###  官方命题

## 二、数据集简介

### 1.数据集说明


```
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```


```
# 查看数据集文件结构
!tree objDataset -L 2
```

    objDataset
    ├── barricade
    │   ├── Annotations
    │   └── JPEGImages
    ├── facemask
    │   ├── Annotations
    │   └── JPEGImages
    ├── fire
    │   ├── Annotations
    │   └── JPEGImages
    ├── MidAutumn
    │   ├── Annotations
    │   └── JPEGImages
    └── roadsign_voc
        ├── Annotations
        └── JPEGImages
    
    15 directories, 0 files


### 2.数据准备


```
# 安装PaddleX
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: paddlex in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.3.11)
    Requirement already satisfied: xlwt in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.3.0)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Requirement already satisfied: shapely>=1.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.7.1)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: pycocotools; platform_system != "Windows" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.0.2)
    Requirement already satisfied: paddlehub==2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.0)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: paddleslim==1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.1.1)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.15.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.2.3)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.16.4)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (7.1.2)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.22.1)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (41.4.0)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (18.1.1)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: paddle2onnx>=0.5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (0.7)
    Requirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.7)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2.4.2)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.10.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (7.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (0.16.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: scipy>=0.17.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.3.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.70.11.1)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.3.3)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (7.2.0)



```
# 划分数据集
!paddlex --split_dataset --format VOC --dataset_dir objDataset/fire --val_value 0.2 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Dataset Split Done.[0m
    [0mTrain samples: 345[0m
    [0mEval samples: 98[0m
    [0mTest samples: 49[0m
    [0mSplit files saved in objDataset/fire[0m
    [0m[0m

### 3.数据的预处理


```
from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])
```


```
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/fire',
    file_list='objDataset/fire/train_list.txt',
    label_list='objDataset/fire/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/barricade',
    file_list='objDataset/fire/val_list.txt',
    label_list='objDataset/barricade/labels.txt',
    transforms=eval_transforms)
```

    2021-08-14 20:14:25 [INFO]	Starting to read file list from dataset...
    2021-08-14 20:14:25 [INFO]	345 samples in file objDataset/fire/train_list.txt
    creating index...
    index created!
    2021-08-14 20:14:25 [INFO]	Starting to read file list from dataset...



    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-12-0e7aadfc7f33> in <module>
         14     file_list='objDataset/fire/val_list.txt',
         15     label_list='objDataset/barricade/labels.txt',
    ---> 16     transforms=eval_transforms)
    

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/datasets/voc.py in __init__(self, data_dir, file_list, label_list, transforms, num_workers, buffer_size, parallel_method, shuffle)
        236 
        237         if not len(self.file_list) > 0:
    --> 238             raise Exception('not found any voc record in %s' % (file_list))
        239         logging.info("{} samples in file {}".format(
        240             len(self.file_list), file_list))


    Exception: not found any voc record in objDataset/fire/val_list.txt


## 三、模型训练


```
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3
# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')
```


```
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')
```

    2021-08-14 20:14:49 [INFO]	Downloading MobileNetV1_pretrained.tar from http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar


    100%|██████████| 16760/16760 [00:00<00:00, 22380.48KB/s]


    2021-08-14 20:14:50 [INFO]	Decompressing output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained.tar...
    2021-08-14 20:14:51 [INFO]	Load pretrain weights from output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained.
    2021-08-14 20:14:51 [INFO]	There are 135 varaibles in output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained are loaded.
    2021-08-14 20:15:18 [INFO]	[TRAIN] Epoch=1/270, Step=2/43, loss=13465.625, lr=0.0, time_each_step=13.37s, eta=44:28:9
    2021-08-14 20:15:39 [INFO]	[TRAIN] Epoch=1/270, Step=4/43, loss=15366.742188, lr=0.0, time_each_step=11.86s, eta=39:25:19
    2021-08-14 20:15:52 [INFO]	[TRAIN] Epoch=1/270, Step=6/43, loss=7559.179688, lr=1e-06, time_each_step=10.15s, eta=33:43:40
    2021-08-14 20:16:02 [INFO]	[TRAIN] Epoch=1/270, Step=8/43, loss=4799.871094, lr=1e-06, time_each_step=8.85s, eta=29:24:28
    2021-08-14 20:16:17 [INFO]	[TRAIN] Epoch=1/270, Step=10/43, loss=3063.631836, lr=1e-06, time_each_step=8.55s, eta=28:24:19
    2021-08-14 20:16:31 [INFO]	[TRAIN] Epoch=1/270, Step=12/43, loss=3719.883545, lr=1e-06, time_each_step=8.34s, eta=27:42:37
    2021-08-14 20:16:53 [INFO]	[TRAIN] Epoch=1/270, Step=14/43, loss=3003.431641, lr=2e-06, time_each_step=8.66s, eta=28:46:34
    2021-08-14 20:17:04 [INFO]	[TRAIN] Epoch=1/270, Step=16/43, loss=886.507874, lr=2e-06, time_each_step=8.29s, eta=27:31:55
    2021-08-14 20:17:16 [INFO]	[TRAIN] Epoch=1/270, Step=18/43, loss=420.128326, lr=2e-06, time_each_step=8.05s, eta=26:44:20
    2021-08-14 20:17:32 [INFO]	[TRAIN] Epoch=1/270, Step=20/43, loss=282.639282, lr=2e-06, time_each_step=8.0s, eta=26:34:49
    2021-08-14 20:17:47 [INFO]	[TRAIN] Epoch=1/270, Step=22/43, loss=291.273865, lr=3e-06, time_each_step=7.43s, eta=24:40:30
    2021-08-14 20:18:05 [INFO]	[TRAIN] Epoch=1/270, Step=24/43, loss=245.878082, lr=3e-06, time_each_step=7.32s, eta=24:17:13


## 四、总结

#### 官方的教学真不错

## 五、个人简介
#### 
