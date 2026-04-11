# An Efficient Gloss-Free Sign Language Translation Using Spatial Configurations and Motion Dynamics with LLMs

Official implementation for the NAACL 2025 [paper](https://aclanthology.org/2025.naacl-long.197.pdf): An Efficient Gloss-Free Sign Language Translation Using Spatial Configurations and Motion Dynamics with LLMs

**This fork (SpaMo-PJM)** extends the original SpaMo framework to support Polish Sign Language (PJM - Polski Język Migowy) from the [croco-corp/pjm-segments](https://huggingface.co/datasets/croco-corp/pjm-segments) dataset.


## Introduction

![model architecture](images/overview.png)

We introduce a novel gloss-free framework, **Spa**tial and **Mo**tion-based Sign Language Translation (**SpaMo**). 
SpaMo is designed to fully exploit the spatial configurations and motion dynamics in sign videos using off-the-shelf visual encoders, without requiring domain-specific fine-tuning.
As shown in the figure above, the core idea is simple: We extract spatial features (representing spatial configurations) and motion features (capturing motion dynamics) using two different visual encoders, then feed these into an LLM with a language prompt.


## Environment

Install dependencies using:
```bash
pip install -r requirements.txt
```


## Data Preparation

We validate our method on three datasets:
- [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)
- [How2Sign](https://how2sign.github.io/)

**This fork additionally supports:**
- [PJM Korpus](https://huggingface.co/datasets/croco-corp/pjm-segments) - Polish Sign Language dataset

### Spatial and Motion Features

SpaMo utilizes two complementary feature types:
1. **Spatial Features**: Extracted with ViT models to capture static visual information
2. **Motion Features**: Extracted with VideoMAE models to capture temporal dynamics

#### Extracting Spatial Features

To extract spatial features using the CLIP ViT model:

```bash
python scripts/vit_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name openai/clip-vit-large-patch14 \
    --video_root /PATH/TO/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
    --cache_dir /PATH/TO/CACHE_DIR \
    --save_dir /PATH/TO/SAVE_DIR \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 32 \
    --device cuda:0
```

Key parameters:
- `--model_name`: CLIP ViT model variant (default: openai/clip-vit-large-patch14)
- `--s2_mode`: Use "s2wrapping" for multi-scale feature extraction
- `--scales`: Scales for multi-scale feature extraction (default: 1 2)

#### Extracting Motion Features

To extract motion features using VideoMAE:

```bash
python scripts/mae_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name MCG-NJU/videomae-large \
    --video_root /PATH/TO/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
    --cache_dir /PATH/TO/CACHE_DIR \
    --save_dir /PATH/TO/SAVE_DIR \
    --overlap_size 8 \
    --batch_size 32 \
    --device cuda:0
```

For convenience, you can download our pre-extracted features from [here](https://www.dropbox.com/scl/fo/vgbws4cftewpoc6kudoap/AOtWs7adP4AvK0iT7KkWaJk?rlkey=nf3wp64zenqx3t2z695ndzcy7&st=9ydialet&dl=0).

#### Extracting Motion Features for PJM Dataset

**This fork includes a specialized script for PJM feature extraction** that works with the HuggingFace dataset:

```bash
python scripts/mae_features_for_pjm.py \
    --dataset-path croco-corp/pjm-segments \
    --crop-params-path crop_params/crop_params.lmdb \
    --model-name MCG-NJU/videomae-large \
    --cache-dir /PATH/TO/CACHE_DIR \
    --save-dir /PATH/TO/SAVE_DIR \
    --overlap-size 8 \
    --batch-size 32 \
    --device cuda:0 \
    --split train
```

**Key differences from the base SpaMo:**
- Loads videos directly from HuggingFace datasets instead of local files
- Uses LMDB database for crop parameters to normalize frame regions
- Supports resumable processing - automatically skips already processed videos
- Outputs features in HDF5 format: `mae_feat_pjm_{split}.h5`
- Includes error logging to `mae_features_extract_errors.log`
- Processes 16-frame video chunks with configurable overlap (default: 8 frames)

**Prerequisites for PJM:**
1. Crop parameters must be stored in LMDB format at `crop_params/crop_params.lmdb`
2. The PJM dataset will be automatically downloaded from HuggingFace
3. Videos are processed as binary MP4 data from the dataset


## Model Training and Evaluation

### Training

Train the SpaMo model with:

```bash
python main.py -c configs/finetune.yaml -e bleu
```

### Evaluation

Evaluate a trained model using:

```bash
python main.py -c configs/finetune.yaml -e bleu --train False --test True --ckpt /PATH/TO/CHECKPOINT
```

Replace `/PATH/TO/CHECKPOINT` with your model checkpoint path.
Pre-trained checkpoints are available for download [here](https://www.dropbox.com/scl/fi/c9khflgxgl96lx919p6oq/spamo.ckpt?rlkey=gp3zmk6jwg9cnf3e2hpw268ih&st=u103orvs&dl=0).


## PJM-Specific Changes

This fork includes several modifications to support Polish Sign Language (PJM):

### Architecture
- **Feature Extraction Pipeline**: Custom implementation for HuggingFace dataset integration
- **Preprocessing**: LMDB-based crop parameter system for consistent frame normalization
- **Data Format**: HDF5 storage with resumable processing capabilities
- **Error Handling**: Comprehensive logging and error recovery during feature extraction

### Key Components

#### 1. PJM Feature Extraction Script
[scripts/mae_features_for_pjm.py](scripts/mae_features_for_pjm.py)
- Extracts VideoMAE motion features from PJM dataset
- Integrates with HuggingFace `datasets` library
- Supports interrupted processing resumption
- Batch processing with configurable overlap

#### 2. PJM Preprocessing Utilities
[utils/pjm/preprocessing.py](utils/pjm/preprocessing.py)
- `ImageConverter`: Applies crop parameters from LMDB to normalize frames
- `get_video_frames()`: Extracts frames from binary MP4 data using PyAV
- Handles aspect ratio preservation and padding

#### 3. Configuration
Project configuration in [pyproject.toml](pyproject.toml):
- Package name: `spamo-pjm`
- Additional dependencies: `h5py`, `lmdb`, `msgpack`, `av`, `datasets`

### Data Pipeline

```
PJM Dataset (HuggingFace)
    ↓
Load binary MP4 videos
    ↓
Extract frames with PyAV
    ↓
Apply crop parameters (LMDB)
    ↓
Process in 16-frame chunks (overlap: 8)
    ↓
Extract VideoMAE features
    ↓
Save to HDF5 with metadata
```

### Technical Details

**Feature Storage Format (HDF5):**
```python
Attributes:
  - model: "MCG-NJU/videomae-large"
  - overlap_size: 8
  - nth_layer: -1
  - dataset_name: "PJM"
  - split: "train"
  - num: <total_videos>

Datasets:
  /<video_id>: array of shape (num_chunks, 1024)
    Attributes:
      - num_chunks: number of temporal chunks
      - features_dim: 1024 (VideoMAE-large output dimension)
```

**Crop Parameters (LMDB):**
- Key: video_id (string)
- Value: MessagePack-encoded crop parameters
- Used for consistent signer region normalization

### Dependencies Added for PJM Support
- `h5py>=3.15.1` - HDF5 file format for feature storage
- `lmdb>=1.7.5` - Crop parameter database
- `msgpack>=1.1.2` - Serialization for crop parameters
- `av>=16.0.1` - Video frame extraction from binary data
- `datasets>=4.4.1` - HuggingFace dataset integration

### Development Timeline
Key commits for PJM support:
- `d3da538` - Dataset class for PJM
- `319445b` - MAE for PJM with custom iterator
- `13c359a` - MAE HDF5 saving logic
- `0dbcf9f` - Resumable processing capability
- `88026aa` - Variable naming improvements
- `b8cdec4` - Error handling enhancements


## Citation

Please cite our works if you find this repo is helpful.

```bash
@inproceedings{hwang2025efficient,
  title={An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs},
  author={Hwang, Eui Jun and Cho, Sukmin and Lee, Junmyeong and Park, Jong C},
  booktitle={NAACL},
  year={2025}
}
```
