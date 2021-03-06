#preprocess.py - written by Justin Wang and Gaurab Banerjee

import os
import librosa
# import librosa.display
import librosa.effects
import librosa.util
# also check you can import numpy and matplotlib, which are part of the␣ 􏰀→anaconda package
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from collections import defaultdict
from time import process_time
import pandas as pd


if __name__ == "__main__":
    wav_dir = 'Audio_Speech_Actors_01-24'
    
    N_FFT = 2048
    FMAX = 4096
    HOP_LENGTH = 512

    data = defaultdict(list)

    start = process_time()
    for i, directory in enumerate(os.listdir(wav_dir)):
        print(f"Actor {i} parsing.")
        if os.path.isdir(os.path.join(wav_dir, directory)):
            for filename in os.listdir(os.path.join(wav_dir, directory)):
                if filename.endswith(".wav"):
                    wav, sr = librosa.load(os.path.join(wav_dir, directory, filename))
                    data['mel'].append(librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=256, fmax=FMAX, n_fft=N_FFT))
                    data['mfcc'].append(librosa.feature.mfcc(wav, hop_length=HOP_LENGTH, n_mfcc=20, fmax=FMAX))
                    data['chromagram'].append(librosa.feature.chroma_stft(wav, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT))
                    data['spec_contrast'].append(librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)) #stft here instead of wav??
                    data['tonnetz'].append(librosa.feature.tonnetz(y=librosa.effects.harmonic(wav), sr=sr))

                    data['filename'].append(filename)

                    # parse filename
                    identifiers = filename.split('-')
                    emotion = int(identifiers[2])
                    intensity = int(identifiers[3])
                    statement = int(identifiers[4])
                    repeat = int(identifiers[5])
                    gender = int(identifiers[6][:2]) % 2 #only want the two digits, not the ".wav"; eg 01 is wanted not 01.wav
                    
                    data['emotion'].append(emotion)
                    data['intensity'].append(intensity)
                    data['statement'].append(statement)
                    data['repeat'].append(repeat)
                    data['gender'].append(gender)
    end = process_time()

    print("elapsed " + str(end-start) + " seconds")
    
#     f = open("preprocessed_data_3_6.pkl", "wb")
#     pickle.dump(data, f)
#     f.close()
    
#     f = open("preprocessed_data_03_6.pkl", "rb")
#     data = pickle.load(f)
#     f.close()
#     print(data.keys())
    
    
    for filename in data['filename']:
        identifiers = filename.split('-')
        emotion = int(identifiers[2])
        intensity = int(identifiers[3])
        statement = int(identifiers[4])
        repeat = int(identifiers[5])
        gender = int(identifiers[6][:-4]) % 2
        data['emotion'].append(emotion)
        data['intensity'].append(intensity)
        data['statement'].append(statement)
        data['repeat'].append(repeat)
        data['gender'].append(gender)

#     f = open("preprocessed_data_02_28.pkl", "wb")
#     pickle.dump(data, f)
#     f.close()
    
    df = pd.concat([
        pd.DataFrame(data['emotion']),
        pd.DataFrame(data['intensity']),
        pd.DataFrame(data['statement']),
        pd.DataFrame(data['repeat']),
        pd.DataFrame(data['gender']), 
        pd.DataFrame(data['mel']),
        pd.DataFrame(data['mfcc']),
        pd.DataFrame(data['chromagram']),
        pd.DataFrame(data['spec_contrast']),
        pd.DataFrame(data['tonnetz']),
        pd.DataFrame(data['filename'])
    ], axis=1)

    df.columns = ['emotion', 'intensity', 'statement', 'repeat', 'gender', 'mel', 'mfcc', 'chromagram', 'spec_contrast', 'tonnetz', 'filename']
    
#     df.head()
    temp = df.dropna()
    print(len(df), len(temp))
    
    #split dataset into train (80%) and val_test (20%) then split val_test half and half for val and test sets
    train, val_test = train_test_split(temp, test_size=0.2, random_state=42, 
                                                        stratify=temp[['emotion', 'intensity', 'statement', 'repeat', 'gender']])
    val, test = train_test_split(val_test, test_size=0.5, random_state=42, 
                                                        stratify=temp[['emotion', 'intensity', 'statement', 'repeat', 'gender']])
#     print(train.head()

    f = open("preprocessed_data_split_nona_03_06.pkl", "wb")
    pickle.dump((train, val, test), f)
    f.close()
    
    #testing if opening fails
    f = open("preprocessed_data_split_nona_02_28.pkl", "rb")
    train, val, test = pickle.load(f)
    f.close()