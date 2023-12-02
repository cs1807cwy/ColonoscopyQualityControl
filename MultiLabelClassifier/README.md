# 多标签多任务分类器

实现基于PyTorch和Lightning框架

[TOC]

## 引用申明

官方CSRA实现请参见：https://github.com/Kevinz-code/CSRA

参考论文"Residual Attention: A Simple but Effective Method for Multi-Label Recognition"：https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Residual_Attention_A_Simple_but_Effective_Method_for_Multi-Label_Recognition_ICCV_2021_paper.html

官方ViT-L实现请参见：https://github.com/google-research/vision_transformer
参考论文"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"：https://iclr.cc/virtual/2021/oral/3458

## 目录结构

```bash
MultiLabelClassifier
├─Network  # 网络模块目录
| ├─ClassifyHead.py  # CSRA模块
│ └─ViT_Backbone.py  # 主干网络（使用Timm迁移模型）
├─DataModule.py  # 数据管理模型
├─Dataset.py  # 数据预处理
├─Modelv2.py  # ViT-L-Patch16-224 CSRA网络模型
└─Modelv3.py  # ViT-L-Patch14-336 CSRA网络模型
```

