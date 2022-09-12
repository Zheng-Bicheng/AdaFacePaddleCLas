## AdaFace数据集使用教程(参考[AdaFace](https://github.com/mk-minchul/AdaFace/blob/master/README_TRAIN.md))
### 名词解释
```text
如果存放数据集的目录结构为:
├── dataset
│   ├── REAME.md
│   ├── mx_recordio_2_images.py
│   └── faces_webface_112x112

则把dataset目录叫做DATASET_ROOT，把faces_webface_112x112目录叫做DATASET_NAME
```

1. 下载[faces_webface_112x112数据集](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
2. 解压到DATASET_ROOT目录
3. PaddleClas根目录下运行脚本命令
    ```
   python dataset/mx_recordio_2_images.py --root_dir ./dataset/faces_webface_112x112/ --output_dir ./dataset/faces_webface_112x112/
   ```