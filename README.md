# 110-1-NTU-DBME5028-Histology-images-query
Final Project: Histology images query (unsupervised)
Kaggle: https://www.kaggle.com/c/histology-images-query-competition/overview

Teem members:\
f09921058 陳羿翔, r09942171 黃繼綸

# Environment
OS : Ubuntu 16.04 \
Language: Python37
Torch: 21.08+

# How to use
### Download dataset in Kaggle
https://www.kaggle.com/c/histology-images-query-competition/data
```bash
$ kaggle competitions download -c histology-images-query-competition
```

### Download the pretrained models
https://drive.google.com/drive/folders/1u2wWuPfb327AHGi8WRCTMiVxmIRPrUHg?usp=sharing


### Training from scratch
```bash
$ python train.py --data path/to/train
```

### Testing
```bash
$ python inference.py --data path/to/test
```
