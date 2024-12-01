import librosa
from tqdm import tqdm
import numpy as np
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

def features_extractor(file):
        audio, sample_rate = librosa.load(file) 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features

def load_data():
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
    
    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())
    if True:
        features = X
        features = np.array(features)
        features = features.reshape((features.shape[0], 1, features.shape[1]))
        X = features
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)
    # traindata = DataLoader(BabyData(X_train,y_train), batch_size=32)
    # testdata = DataLoader(BabyData(X_test,y_test))
    # return traindata, testdata
    return (X_train,y_train),(X_test,y_test)

def load_model(model: keras.Model, filepath: str):
    """
    Load weights into a TensorFlow model from a .npz file.

    Args:
        model (keras.Model): The TensorFlow model to load weights into.
        filepath (str): Path to the .npz file containing weights.
    """
    # Load the saved weights as a list of numpy arrays
    with np.load(filepath) as data:
        weights = [data[key] for key in sorted(data.files)]  # Extract weights in order

    # Set the weights to the model
    model.set_weights(weights)
    print(f"Model weights loaded from {filepath}")
        