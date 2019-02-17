rmSimilarImg
====================

## Overview

A script that deletes similar images through vectorization and clustering of images.

The following options are avalidable for vectorization:

- Middle layer output of [Xception by keras-application](https://keras.io/ja/applications/#xception)
- [wavelet hash by Imagehash](https://github.com/JohannesBuchner/imagehash)

[Clustering by KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

## Requirement
```
python >= 3.6.4
pip install -r requairements.txt
```

## Usage
```sh
python main.py INPUT_DIR OUTPUT_DIR NUM_SAMPLE 
            [-h] [--extractor {Xception,whash}]
```