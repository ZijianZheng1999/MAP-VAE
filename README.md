# MAP-VAE
A Modular Alignment and Pre-trained Framework for Multi-modal Variational Inference

## Requirements

* ```tqdm```: no version requirement
* ```pandas```: no version requirement
* ```pytorch```: 2.3.1 (recommend)
* ```torchvision```: 0.18.1 (recommend)
* ```torchaudio```: 2.3.1 (recommend)
* ```numpy```: 2.0 (required)
* ```matplotlib```: 3.7.1 (required)
* ```pytorch3d```: 0.7.8 (required)

## Dataset

* KITTI

  * link: https://www.cvlibs.net/datasets/kitti/

  * To make the loader work properly, please organize the dataset in the following directory:

    ``````
    <kitti>
       └─raw
           ├─testing
           │  ├─calib
           │  ├─image_2
           │  └─velodyne
           └─training
               ├─calib
               ├─image_2
               ├─label_2
               └─velodyne
    ``````

* mHealth

  * link: https://archive.ics.uci.edu/dataset/319/mhealth+dataset

  * To make the loader work properly, please organize the dataset in the following directory:

    ```
    <mHealth>
       └─mHealth_subject1.log
       ├─mHealth_subject2.log
       ├─ .
       ├─ .
       ├─ .
       └─mHealth_subject10.log
    ```

* URFall

  * link: https://fenix.ur.edu.pl/~mkepski/ds/uf.html

  * To make the loader work properly, please organize the dataset in the following directory:

    ```
    <URFall>
       ├─ADL_sequences
       │  ├─1
       │  │ ├─adl-01-acc.csv
       │  │ ├─adl-01-data.csv
       │  │ ├─adl-01-cam0-d
       │  │ └─adl-01-cam0-rgb
       │  ├─ .
       │  ├─ .
       │  ├─ .
       │  ├─40
       │  │  ├─adl-40-acc.csv
       │  │  ├─adl-40-data.csv
       │  │  ├─adl-40-cam0-d
       │  │  └─adl-40-cam0-rgb
       └─Fall_sequences
           ├─1
           │ ├─fall-01-acc.csv
           │ ├─fall-01-data.csv
           │ ├─fall-01-cam0-d
           │ ├─fall-01-cam0-rgb
           │ ├─fall-01-cam1-d
           │ └─fall-01-cam1-rgb
           ├─ .
           ├─ .
           ├─ .
           └─30
              ├─fall-30-acc.csv
              ├─fall-30-data.csv
              ├─fall-30-cam0-d
              ├─fall-30-cam0-rgb
              ├─fall-30-cam1-d
              └─fall-30-cam1-rgb
    ```
