# Reading-VQA

Project of the 14th XingHuo Project Election

### Requirements

* python3
* torch
* torchvision
* nltk
* tqdm
* h5py
* transformers

### Usage

```bash
usage: main.py [-h] --data_root DATA_ROOT [--stage {train,test}] [--cuda] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--load LOAD][--save_dir SAVE_DIR] [--save_freq SAVE_FREQ]

Reading-VQA

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        root directory of dataset
  --stage {train,test}  model stage
  --cuda                enable cuda
  --epochs EPOCHS       total epochs to train
  --batch_size BATCH_SIZE
                        mini-batch size (default: 64)
  --lr LR               learning rate
  --load LOAD           loading specific model checkpoint
  --save_dir SAVE_DIR   directory for saving model checkpoints
  --save_freq SAVE_FREQ
                        number of iterations between two saving actions
```

```bash
# Examples
# 开始训练
python main.py --root=/data/VisualGenome --stage=train --cuda
# 从已有 checkpoint 恢复
python main.py --root=/data/VisualGenome --stage=train --cuda --load=./checkpoints/xxx.pt
```

### 准备数据集

本项目的数据集为 Visual Genome，我们在使用时进行了一定的整理。

我们按照 80% - 10% - 10% 的比例划分训练 / 开发 / 测试集。

其中关键数据文件 `data.json` 格式如下：
```json
{
    "train": {
        "images": {
            "[image_id]": {
                "index": "[index_in_image_data.json]",
                "path": "[local_image_path]",
                "desc": [
                    "[description sentence 1]",
                    "[description sentence 2]",
                    "..."
                ]
            }
        },
        "qas": [
            {
                "image_id": "[number]",
                "question": "[question?]",
                "answer": "[answer]"
            }
        ]
    },
    "dev": {
        ...
    },
    "test": {
        ...
    }
}
```

### 预提取图片特征

由于我们采用的图片特征提取 CNN (ResNet152) 在训练时会占用较多的 GPU 显存(batch_size 将受到限制) 以及增加额外的计算时间，因此我们预提取了数据集 (Visual Genome) 的图片特征，用 hdf5 格式保存在数据目录下。

```python
# run `python scripts/extract.py --help` for more information
python scripts/extract.py --root=/path/to/your_data_root --cuda
```



#### Reference 

1. [Ben-Younes, Hedi, et al. "Mutan: Multimodal tucker fusion for visual question answering." Proceedings of the IEEE international conference on computer vision. 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ben-younes_MUTAN_Multimodal_Tucker_ICCV_2017_paper.pdf)
2. [Antol, Stanislaw, et al. "Vqa: Visual question answering." Proceedings of the IEEE international conference on computer vision. 2015.](http://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)



#### Acknowledgement

My advisor is Professor Xiaolin Hu.

This Project is modified and extended based on "Introduction to Deep Learning" course project. 

My teammates are Bohan Chen (@acyume), Zhuoer Feng

