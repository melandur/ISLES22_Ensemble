# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
import warnings

# Suppress third-party library warnings - set up before any imports
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ignite')
warnings.filterwarnings('ignore', category=UserWarning, module='monai.utils.module')
warnings.filterwarnings('ignore', message='.*TorchScript.*functional optimizers.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*deprecated.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*deprecated.*')

# Also set environment variable to suppress warnings in subprocesses
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:ignite,ignore::UserWarning:monai.utils.module'

ENSEMBLE_PATH = os.getcwd()  # path-to-ensemble-repo
sys.path.append(ENSEMBLE_PATH)
from src.isles22_ensemble import IslesEnsemble

# .nii/.nii.gz/.mha or DICOM folder
INPUT_FLAIR = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_flair.nii.gz')  # path-to-FLAIR
INPUT_DWI = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_dwi.nii.gz')      # pat-t-DWI
INPUT_ADC = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_adc.nii.gz')      # path-to-ADC
OUTPUT_PATH = os.path.join(ENSEMBLE_PATH, 'example_test')                                      # path-to-output

stroke_segm = IslesEnsemble()

# Run the ensemble prediction
stroke_segm.predict_ensemble(ensemble_path=ENSEMBLE_PATH,
                             input_dwi_path=INPUT_DWI,
                             input_adc_path=INPUT_ADC,
                             input_flair_path=INPUT_FLAIR,
                             output_path=OUTPUT_PATH)

