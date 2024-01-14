# CrashCar101: Procedural Generation for Damage Assessment
Created by [Jens Parslov](https://www.linkedin.com/in/jens-parslov/), [Erik Riise](https://www.linkedin.com/in/erik-riise-97a6a31a6/), [Dim P. Papadopoulos](https://orbit.dtu.dk/en/persons/dimitrios-papadopoulos)
[[`Paper (WACV 2024)`](https://openaccess.thecvf.com/content/WACV2024/papers/Parslov_CrashCar101_Procedural_Generation_for_Damage_Assessment_WACV_2024_paper.pdf)] [[`Project page`](https://crashcar.compute.dtu.dk)]

Welcome to CrashCar! This repository contains the procedural generation pipeline used to generate [CrashCar101](https://www.crashcar.compute.dtu.dk).

## Installation
To use the CrashCar data, you can simply clone this repository to your local machine:
The code is only dependent on the Blenders internal python interpreter, you just need to download [Blender 3.3](https://www.blender.org/download/lts/3-3/)

## Car models and HDRI setup
1. Download the shapenet dataset from [The shapenet project page](https://shapenet.org/download/shapenetcore) and place all the contents of the directory named "02691156" into `data/shapenet`.
2. Download HDRIs and place them into `data/HDRI` from [Polyhaven](https://polyhaven.com) or using [this tool](https://github.com/theadisingh/HDRI-Haven-Downloader).


## Generate images
Simply run the command
`<path to blender executable> main.blend --background --python main.py`

## Citation
```bibtex
@InProceedings{parslov_2024_WACV,
        author    = {Parslov, Jens and Riise, Erik and Papadopoulos, Dim P.},
        title     = {CrashCar101: Procedural Generation for Damage Assessment},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month     = {January},
        year      = {2024},
    }
```