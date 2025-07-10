---
layout: post
title: "DeepKe官方Demo"
date:   2025-7-10
tags: [大模型]
comments: true
author: tom
---

<!-- more -->

## 设置Github镜像

```bash
git config --system url."https://githubfast.com/".insteadOf https://github.com/

如果要取消，则输入：
git config --system --unset url.https://githubfast.com/.insteadof
```

## 创建conda环境

```bash
conda create -n deepke python=3.8
conda activate deepke

# 安装torch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 使用阿里云镜像安装torch 1.11.0
pip install https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl https://mirrors.aliyun.com/pytorch-wheels/cu113/torchvision-0.12.0+cu113-cp38-cp38-linux_x86_64.whl https://mirrors.aliyun.com/pytorch-wheels/cu113/torchaudio-0.11.0+cu113-cp38-cp38-linux_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple/
```

## 安装DeepKE

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE

pip install pip==24.0

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python setup.py install
python setup.py develop

pip install prettytable==2.4.0
pip install ipython==8.12.0
```

## 下载数据集

```bash
cd example/ner/standard
wget 121.41.117.246:8080/Data/ner/standard/data.tar.gz
tar -xzvf data.tar.gz
```

## 配置wandb

在 https://wandb.ai/ 上注册账号，并新建一个project，取一个名字，比如：`deepke-ner-official-demo`

打开 https://wandb.ai/authorize 获取 API key

运行 `wandb init`，输入刚获取的 API key 和创建的project

## 运行训练和预测

删除之前训练时保存的checkpoints和logs文件夹（如果有）

```bash
rm -r checkpoints/
rm -r logs/
```

### lstm_crf

打开 `example/ner/standard/run_lstmcrf.py`， 确保wandb和yaml库有正常导入

```python
import wandb
import yaml
```

修改wandb的project名称

```python
if config['use_wandb']:
    wandb.init(project="deepke-ner-official-demo")
```

修改 `example/ner/standard/conf/config.yaml` 中的 `use_wandb` 为 `True`。

如果需要使用多个GPU训练，修改 `example/ner/standard/conf/train.yaml` 中的 `use_multi_gpu` 为 `True`

开始训练：

```python
python run_lstmcrf.py

>> total: 109870 loss: 27.181508426008552
precision    recall  f1-score   support

B-LOC     0.8920    0.8426    0.8666      1951
B-ORG     0.8170    0.7439    0.7787       984
B-PER     0.8783    0.8167    0.8464       884
I-LOC     0.8650    0.8264    0.8453      2581
I-ORG     0.8483    0.8365    0.8424      3945
I-PER     0.8860    0.8436    0.8643      1714
O     0.9861    0.9912    0.9886     97811

accuracy                         0.9732    109870
macro avg     0.8818    0.8430    0.8618    109870
weighted avg     0.9727    0.9732    0.9729    109870
```

用于的预测文本保存在`example/ner/standard/conf/predict.yaml`中，修改为如下：

```python
text: "“热水器等以旧换新，节省了2000多元。”10月3日，在湖北省襄阳市的一家购物广场，市民金煜轻触手机，下单、付款、登记。湖北着力推动大规模设备更新和消费品以旧换新。“力争到今年底，全省汽车报废更新、置换更新分别达到4.5万辆、12.5万辆，家电以旧换新170万套。”湖北省商务厅厅长龙小红介绍。"
```

运行预测：

```python
python predict.py
```

NER结果

```python
[('湖', 'B-LOC'), ('北', 'I-LOC'), ('省', 'I-LOC'), ('襄', 'B-LOC'), ('阳', 'I-LOC'), ('市', 'I-LOC'), ('场', 'I-LOC'), ('煜', 'I-PER'), ('湖', 'B-ORG'), ('北', 'I-ORG'), ('省', 'I-ORG'), ('商', 'I-ORG'), ('务', 'I-ORG'), ('厅', 'I-ORG'), ('厅', 'I-ORG'), ('龙', 'B-PER'), ('小', 'I-PER'), ('红', 'I-PER')]
```

### bert

修改 `example/ner/standard/conf/config.yaml`中的`hydra/model`为`bert`。

bert的超参设置在 `example/ner/standard/conf/hydra/model/bert.yaml`，如有需要可以修改。

修改 `example/ner/standard/conf/config.yaml` 中的 `use_wandb` 为 `True`。

修改 `example/ner/standard/run_bert.py` 中的wandb的project名称：

```python
if cfg.use_wandb:
    wandb.init(project="deepke-ner-official-demo")
```

根据需要，修改`example/ner/standard/conf/train.yaml`中的`train_batch_size`，对于bert来说推荐不小于64

开始训练：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python run_bert.py
```

### w2ner

w2ner是一个新的SOTA模型。

基于**W2NER** (AAAI’22)的应对多种场景的实体识别方法 (详情请查阅论文[Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/pdf/2112.10070.pdf)).

命名实体识别 (NER) 涉及三种主要类型，包括平面、重叠（又名嵌套）和不连续的 NER，它们大多是单独研究的。最近，人们对统一 NER 越来越感兴趣， `W2NER`使用一个模型同时处理上述三项工作。

由于使用单卡GPU，修改`example/ner/standard/w2ner/conf/train.yaml`中的 `device` 为 `0`。

修改`example/ner/standard/w2ner/conf/train.yaml`中的`data_dir`和`do_train`：

```python
data_dir: "../data"
do_train: True
```

以便使用之前下载的数据集和开始训练。

运行训练：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python run.py
```

