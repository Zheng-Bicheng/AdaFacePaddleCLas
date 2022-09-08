# AdaFacePaddleCLas

## 简介
这个仓库基于Paddle框架，详细训练过程请查看[飞桨AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4479879?contributionType=1&sUid=790375&shared=1&ts=1662618030504)，每天送8小时GPU算力哦


## 更新日志
v0.0 -> 2022-09-08

现在已经可以成功训练AdaFace了

更新内容:
* 修改PaddleClas读取数据集代码
* 修改PaddleClas生成mem代码
* 删除模型，现在将使用百度网盘的方式上传模型

V0.1 -> 2022-09-09

更新内容:
* 从[insightface](https://github.com/deepinsight/insightface/blob/d4d4531a702e22cc7666cda2de6db53f4dc2e4db/recognition/arcface_paddle/datasets/common_dataset.py)上找到了从数据集中提取图片的方式，现在只提取训练集图片，需要注意的是这个脚本不能在AIStudio上运行，如果你需要从头下载数据集，请从我的[github]()仓库中下载源码，并参考dataset/README.md文档进行操作
* 重写了验证集读取5个数据集的代码，现在能直接从bin文件生成mem文件
* 数据集重新上传了一个小型化的数据集，方便在AIStudio中解压，同时占用更少的内存空间
* 缩减了需要的requirements
* 修复了V0.0版本验证时精度偏小的问题