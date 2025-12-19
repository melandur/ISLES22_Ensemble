![alt text](https://github.com/Tabrisrei/ISLES22_Ensemble/blob/master/ensemble-logo.png)

# DeepIsles
## ISLES'22 Ischemic Stroke Lesion Segmentation  

## Introduction
Algorithm to predict ischemic stroke lesions from MRI data.

This repository contains an ensemble algorithm devised in the context of the ISLES'22 MICCAI Challenge (https://isles22.grand-challenge.org/).
The top-3 leading algorithms are used in a majority-voting scheme to predict outputs.

1. [Installation](#installation)
2. [Usage](#usage)
3. [Get started](#get-started)
4. [Citation](#citation)
5. [About the Ensembled Algorithms](#about-the-ensembled-algorithms)
6. [Questions](#questions)
7. [Acknowledgement](#acknowledgement)


## Installation
1.1) Clone this repository.

```bash
git clone https://github.com/Tabrisrei/ISLES22_Ensemble.git
cd ISLES22_Ensemble
```

1.2) Create a conda environment and install dependencies. 
**Note: Python 3.10, 3.11, or 3.12 is required. Python 3.12 is recommended.**

**Option A: Automated setup (recommended)**
```bash
# Default CUDA 11.8
./setup_env.sh

# Or specify CUDA version (11.8, 12, or 13)
./setup_env.sh 12
```

**Option B: Manual installation**
```bash
conda create --name isles_ensemble python=3.12 pip -y
conda activate isles_ensemble
pip install uv
uv pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch>=2.1.2 torchvision torchaudio
uv pip install -e .
```

**CUDA Version Support:**
- CUDA 11.8 (default): `./setup_env.sh 11.8` or `./setup_env.sh`
- CUDA 12: `./setup_env.sh 12`
- CUDA 13: `./setup_env.sh 13`

If successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

1.3) Download the model weights from [here](https://zenodo.org/records/14026715) and decompress the file inside this repo.
Your directory should look like:
```

ISLES22_Ensemble/
├── weights/
│   ├── SEALS/
│   │   └── (...)
│   ├── NVAUTO/
│   │   └── (...)
│   └── FACTORIZER/
│       └── (...)
```


## Usage
### Supported Formats
- **Input formats**:  `.dcm`, `.nii`, `.nii.gz`, `.mha`.
- **Processing**: The algorithm works directly in the native image space — no additional preprocessing required.

### Required Image Modalities
- **DWI (b=1000)**: Required
- **ADC**: Required
- **FLAIR**: Required for ensemble (optional for single algorithm outputs)


### Python

```bash
ENSEMBLE_PATH = 'path-to-isles-ensemble-repo' 
import sys
from isles22_ensemble import IslesEnsemble
sys.path.append(ENSEMBLE_PATH)

INPUT_FLAIR = 'path-to-flair.nii.gz'
INPUT_ADC = 'path-to-adc.nii.gz'
INPUT_DWI = 'path-to-dwi.nii.gz'
OUTPUT_PATH = 'path-to-output-folder'

stroke_segm = IslesEnsemble()
stroke_segm.predict_ensemble(ensemble_path=ENSEMBLE_PATH,
                 input_dwi_path=INPUT_DWI,
                 input_adc_path=INPUT_ADC,
                 input_flair_path=INPUT_FLAIR,
                 output_path=OUTPUT_PATH)
```

### Docker

Requirements: 
- [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Download](https://hub.docker.com/repository/docker/isleschallenge/deepisles/general) the Docker image or build it as  ```bash docker build -t deepisles .```

Docker usage: 
```bash
docker run --gpus all -v /*path*/ISLES22_Ensemble/data:/app/data deepisles --dwi_file_name sub-strokecase0001_ses-0001_dwi.nii.gz --adc_file_name sub-strokecase0001_ses-0001_adc.nii.gz --flair_file_name sub-strokecase0001_ses-0001_flair.nii.gz
```

### Extra Parameters

- **`skull_strip`**: `True`/`False` (default: `False`) — Perform skull stripping on input images.
- **`fast`**: `True`/`False` (default: `False`) — Run a single model for faster execution.
- **`parallelize`**: `True`/`False` (default: `True`) — Up to 50% faster inference on GPUs with ≥12 GB memory.
- **`save_team_outputs`**: `True`/`False` (default: `False`) — Save outputs of individual models before ensembling.
- **`results_mni`**: `True`/`False` (default: `False`) — Save images and outputs in MNI.



## Get started 
Try DeepIsles out over the provided example data:
```bash
 python scripts/predict.py
```

The example scan belongs to the ISLES'22 dataset (Hernandez Petzsche et al., Sci Data 2022).

## Citation
If you use this repository, please cite the following publications:

1. **de la Rosa, E., Reyes, M., Liew, S. L., Hutton, A., Wiest, R., Kaesmacher, J., ... & Wiestler, B. (2024).**  
   *A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge.*  
   arXiv preprint: [arXiv:2403.19425](https://arxiv.org/abs/2403.19425)

2. **Hernandez Petzsche, M. R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela, W., Reyes, M., ... & Kirschke, J. S. (2022).**  
   *ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.*  
   *Scientific Data, 9*(1), 762.



## About the Ensembled Algorithms 
* Algorithm SEALS is based on [nnUnet](https://github.com/MIC-DKFZ/nnUNet). Git [repo](https://github.com/Tabrisrei/ISLES22_SEALS) 

* Algorithm NVAUTO is based on [MONAI](https://github.com/Project-MONAI/MONAI) Auto3dseg. Git [repo](https://github.com/mahfuzmohammad/isles22)

* Algorithm SWAN is based on [FACTORIZER](https://github.com/pashtari/factorizer). Git [repo](https://github.com/pashtari/factorizer-isles22)


## Questions
Please contact Ezequiel de la Rosa (ezequiel.delarosa@uzh.ch).

## Acknowledgement
- We thank all ISLES'22 challenge participants, collaborators and organizers for allowing this work to happen. We also thank all developers and maintaners of the repos herein used. 
- Skull-stripping is done with [HD-BET](https://github.com/MIC-DKFZ/HD-BET).
- The used FLAIR-MNI atlas is obtained from [this paper](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.00208/full) (https://zenodo.org/records/3379848). 
