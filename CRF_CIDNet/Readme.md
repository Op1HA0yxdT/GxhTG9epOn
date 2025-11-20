# CRF CIDNet

## 1. Setup

### Dependencies and Installation

- Python 3.7.0
- Pytorch 1.13.1

(1) Create Conda Environment

```bash
conda create --name CIDNet python=3.7.0
conda activate CIDNet
```

(2) Install Dependencies

```bash
pip install -r requirements.txt
```


### Data Preparation


LOL-v1 [Proton Drive](https://drive.proton.me/urls/7HMEGKRD9G#4JGNUqLuigWG)

LOL-v2 (Real and Synthetic) [Proton Drive](https://drive.proton.me/urls/8VX7YB2R9M#yj7WRJj38G9l)


<details close> <summary>datasets (click to expand)</summary>

```
├── datasets
	├── DICM
	├── FiveK
		├── test
			├──input
			├──target
		├── train
			├──input
			├──target
	├── LIME
	├── LOLdataset
		├── our485
			├──low
			├──high
		├── eval15
			├──low
			├──high
	├── LOLv2
		├── Real_captured
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
		├── Synthetic
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
	├── LOL_blur
		├── eval
			├── high_sharp_scaled
			├── low_blur
		├── test
			├── high_sharp_scaled
				├── 0012
				├── 0017
				...
			├── low_blur
				├── 0012
				├── 0017
				...
		├── train
			├── high_sharp_scaled
				├── 0000
				├── 0001
				...
			├── low_blur
				├── 0000
				├── 0001
				...
	├── MEF
	├── NPE
	├── SICE
		├── Dataset
			├── eval
				├── target
				├── test
			├── label
			├── train
				├── 1
				├── 2
				...
		├── SICE_Grad
		├── SICE_Mix
		├── SICE_Reshape
	├── Sony_total_dark
		├── eval
			├── long
			├── short
		├── test
			├── long
				├── 10003
				├── 10006
				...
			├── short
				├── 10003
				├── 10006
				...
		├── train
			├── long
				├── 00001
				├── 00002
				...
			├── short
				├── 00001
				├── 00002
				...
	├── VV
```
</details>

## 2. Testing

Due to how the codebase saves weights, we are unable to provide best checkpoints. However, training CRF-CIDNet will produce very similar results to the ones reported in the paper.

## 3. Training

First, select the CRF-CIDNet variant by using either the CRF `net/CIDNet_CSE_ott.py` (one-two-two : 1-2-2) config, or the CRF-L `net/CIDNet_CSE_tff.py` (two-four-four : 2-4-4) config. To use either, please copy-paste its contents onto `net/CIDNet.py`.

Adjust info in `data/options.py` to set up training parameters.

Moreover, if training on LOL-v2-real, please un-comment the corresponding loss weights in `data/options.py`. For other datasets, use the standard loss weights. 

```bash

python train.py 
```

## 4. Additional Info

Please visit the official CIDNet GitHub page for detailed info on how to navigate the codebase. Our main contribution lies within the `net/CIDNet_CSE_ott.py` and `net/CIDNet_CSE_tff.py`, with few adjustments in other scripts to ensure compatibility.
