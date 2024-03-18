# JPC - Cross-domain Adaptive Crowd Counting

---
This repo is the implementation of [paper]: **undetermined**. The code is developed based on [NLT](https://github.com/taohan10200/NLT).

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Training](#training)
* [Test](#test)
# Getting Started

## Installation
It is recommended to prepare the following dependencies before training.

-  Prerequisites
    - Python 3.7
    - Pytorch >=1.5: http://pytorch.org .
    - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.
-  Code
    - Clone this repository in the directory (```Root/JPC```):
        ```bash
        git clone https://github.com/jax0619/JPC.git
        ```
- Dataset downloading
    - the source dataset: GCC [[Link1](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/Eo4L82dALJFDvUdy8rBm6B0BuQk6n5akJaN1WUF1BAeKUA?e=ge2cRg)] [[Link2](https://v2.fangcloud.com/share/4625d2bfa9427708060b5a5981)] [[Link3](https://pan.baidu.com/s/1OtKqmw84TFbxAiN0H2xBtQ) (pwd:**utdo**)] 
    - other target datasets: 
    ShanghaiTech Part [[Link1](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)] [[Link2](https://pan.baidu.com/s/1nuAYslz)]);
    UCF-QNRF [[Homepage](https://www.crcv.ucf.edu/data/ucf-qnrf/)] [[Download](https://drive.google.com/open?id=1fLZdOsOXlv2muNB_bXEW6t-IS9MRziL6)];
    MALL [[Homepage](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)];
    - Generate the density map with the scripts in `Root/JPC/data_split/generate_den_map` 

## Project Architecture

  - Finally, the folder tree is below:
 ```
    -- ProcessedData
		|-- performed_bak_lite   # this is GCC dataset, which contains 100 scenes (a total of 400 folders).
            |-- scene_00_0
            |	   |-- pngs_544_960
            |	   |-- den_maps_k15_s4_544_960
            |-- ...
            |-- scene_99_3
            |	   |-- pngs_544_960
            |	   |-- den_maps_k15_s4_544_960
    	|-- SHHB
    	    |-- train
    	    |    |-- img
    	    |    |-- den
    	    |-- test
    	    |    |-- img
    	    |    |-- den
    	|-- ...		
	-- JPC
	  |-- data_split
	  |-- dataloader
	  |-- models
	  |-- ...
 ```

### Pre_train on GCC dataset


#### pre_train 
 modify the `__C.phase='pre_train'` in `config.py`, and then run: 
```bash
python pre_train.py
```

## Training

### Train ours JPC

Modify the flowing configurations in `config.py`:
 ```bash
__C.model_type = 'vgg16'
__C.phase = 'train'  # choices=['train','test','pre_train','SE_pre_train','cross_train']
__C.target_dataset = 'SHHA'# dataset choices =  ['SHHB', 'QNRF', 'MALL', 'SHHA']
__C.init_weights ='./' #path of pre-trained weights

```
Then, run the command:
```bash
python da.py
```

## Test
 To evaluate the metrics (MAE\MSE\PNSR\SSIM) on test set, you should fill the model path (`cc_path`) and dataset name in `test.py`, and then run:

 ```bash
python test.py
```

The visual density map can be selectively generated in `Root/JPC/visual-display`.

