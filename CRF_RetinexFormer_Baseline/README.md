# CRF Baseline & CRF RetinexFormer


## 1. Create Environment


- Make Conda Environment
```
conda create -n torch2 python=3.9 -y
conda activate torch2
```

- Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm python_msssim ptflops
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```


## 2. Prepare Dataset
Download the following datasets: 

LOL-v1 [Proton Drive](https://drive.proton.me/urls/7HMEGKRD9G#4JGNUqLuigWG)

LOL-v2 (Real and Synthetic) [Proton Drive](https://drive.proton.me/urls/8VX7YB2R9M#yj7WRJj38G9l)

SDSD [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
   
```

</details>

                


## 3. Testing

Download our pre-trained models [Proton Drive](https://drive.proton.me/urls/5T7FP3GF1C#lEonEnbXGFcC).

For picking models, please check `basicsr/models/archs/RetinexFormer_arch.py` (used as wrapper for Baseline, Baseline+CRF, and RetinexFormer; please follow instructions in file) and `basicsr/models/archs/CSE_RetinexFormer_arch.py` (hosting RetinexFormer-CRF). Note the name transition from CSE (in code) -> CRF (in paper).

Please ensure options file matches selected architecture.

```shell

python Enhancement/test_from_dataset.py --opt path/to/opt.yml  --weights path/to/weights.pth --dataset dataset_name --GT_mean


```



- #### Parameter / FLOPs test.


```shell
python basicsr/complexity.py --opt path/to/opt.yml --warmup 5 --runs 20 --device cuda --resolutions 256x256
```


&nbsp;


## 4. Training


```shell

python3 basicsr/train.py --opt path/to/opt.yml


```
