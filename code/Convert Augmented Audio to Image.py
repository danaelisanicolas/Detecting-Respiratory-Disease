import librosa as lb
import soundfile as sf
from librosa.display import specshow

import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

categories = ['BronchialDeformity', 'COPD', 'Healthy', 'Pneumonia', 'RTI']

dest_loc = './dana/capstone/data/Respiratory_Files/Audio-images/'
source_loc = './dana/capstone/data/Respiratory_Files/New_Audio/'

train_ = 'train/'
test_ = 'val/'

def is_wav(filename):
    '''
        Checks if files are .wav files
        Utility tool in converting wav to png files
    '''
    return filename.split('.')[-1] == 'wav'

#create images using librosa spectogram
def convert_to_spec_image(source_loc, filename, category, save_directory=dest_loc, is_train=False, verbose=False):
    ''' 
        Converts audio file to spec image
        Input file includes path
        Saves the file to a png image in the save_directory
    '''    
    loc = source_loc + train_ + category + '/' + filename
    if is_train == False:
        loc = source_loc + test_ + category + '/' + filename

    if verbose == True:
        print('reading and converting ' + filename + '...')
        
    y, sr = sf.read(loc)

    #Plot signal in
    plt.figure(figsize=(10,3))
    src_ft = lb.stft(y)
    src_db = lb.amplitude_to_db(abs(src_ft))
    specshow(src_db, sr=sr, x_axis='time', y_axis='hz')  
    plt.ylim(0, 5000)
    
    filename_img = filename.split('.wav')[0]
    save_loc = save_directory + train_ + category + '/' + filename_img + '.png'
    if is_train == False:
        save_loc = save_directory + test_ + category + '/' + filename_img + '.png'
        
    plt.savefig(save_loc)
    
    if verbose == True:
        print(filename + ' converted!')
        
    plt.close()

splits = ['train', 'val']

for split in splits:
    for cat in categories:

        print('-' * 100)
        print('working on ' + cat + '...')
        print('-' * 100)

        files = [f for f in listdir(source_loc + split + '/' + cat + '/') 
                 if isfile(join(source_loc + split + '/' + cat + '/', f))]
        for f in files:
            if is_wav(f) == True:
                convert_to_spec_image(source_loc = source_loc, category=cat, filename=f, 
                                      is_train=(split == 'train'), verbose=True)