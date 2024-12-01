import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import librosa
import librosa.display

import os
from tqdm import tqdm
import pathlib

def features_extractor(file):
    audio, sample_rate = librosa.load(file) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

class_indexes = {
    "Full_asphyxia": 0,
    "Full_deaf": 1,
    "Full_hunger-mexicon": 2,
    'Full_normal':3
}

working_dir = 'data/'

sub_dirs = ['Full_asphyxia','Full_deaf','Full_hunger-mexicon','Full_normal']
sub_dirs  = [(sub_dir, working_dir+sub_dir) for sub_dir in sub_dirs]
print(sub_dirs)
audios = []
for path in sub_dirs:
    for file in pathlib.Path(path[1]+'/').iterdir():
        audios.append((class_indexes[path[0]],file)) 

extracted_features=[]
for audio in tqdm(audios):
    file_name = audio[1]
    final_class_labels=audio[0]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


