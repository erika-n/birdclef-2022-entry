from sympy import fu
import torch
from torch.utils.data import Dataset
import torchaudio
import math

import os
from os import listdir
from os.path import isfile, join


import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader,random_split

from torchvision.transforms import Lambda

import json
from timeit import default_timer as timer

# Dataset for BirdClef 2022. 

class BirdClefDataset(Dataset):
    def __init__(self, n_samples=10000, n_per_file = 10, metadata_file="train_metadata_updated.csv", audio_dir="G:/birdclef-2022/train_audio", scored_birds_file="current_birds.json", transform=None, target_transform=None):
        
        scored_birds = ""
        with open(scored_birds_file) as f:
            scored_birds = json.load(f)
        #scored_birds = scored_birds[:4]

        self.metadata_file = pd.read_csv(metadata_file)

        taxonomy_file = "G:/birdclef-2022/eBird_Taxonomy_v2021.csv"
        td = pd.read_csv(taxonomy_file)
        self.metadata_file = pd.merge(self.metadata_file, td, left_on="primary_label", right_on="SPECIES_CODE")


        self.metadata_file = self.metadata_file.loc[self.metadata_file["primary_label"].isin(scored_birds)] # take only from scored birds
        self.metadata_file = self.metadata_file.loc[self.metadata_file["samples"] > n_samples*n_per_file] # take only if there is enough audio for n_per_file examples
        self.metadata_file = self.metadata_file.loc[self.metadata_file["rating"] > 4] # take only highly rated tracks
        self.metadata_file = self.metadata_file.groupby("primary_label").head(40) # take first N from each species
        self.metadata_file.sort_values(by=['primary_label'], inplace=True)
        self.metadata_file = self.metadata_file.reset_index() 


        

        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels = sorted(scored_birds)
        self.n_samples = n_samples
        self.n_per_file = n_per_file

        self.audio_data = []

        for idx, row in self.metadata_file.iterrows():
            audio_path = os.path.join(self.audio_dir, self.metadata_file["filename"][idx])
            audio, rate = torchaudio.load(audio_path)
            audio = audio[0] # use only left channel
            audio = audio[:n_samples*n_per_file] # keep only the bytes used
            self.audio_data.append(audio)
            # print("adding audio data ", idx)
            # print("filename", audio_path)






    def __len__(self):
        return len(self.metadata_file)*self.n_per_file

    def __getitem__(self, idx):

        which_n = math.floor(idx/(len(self.metadata_file)))
        which_file = idx % len(self.metadata_file)
        # print("len file", len(self.metadata_file))
        # print("idx",  idx)
        # print("which_file", which_file)
        # print("len audio data", len(self.audio_data))
      
        audio = self.audio_data[which_file]
        audio = audio[which_n*self.n_samples:(which_n + 1)*self.n_samples] # use the first N bytes
        audio = audio.reshape([1, -1]) # add channel field back

        text_label = self.metadata_file["primary_label"][which_file]
        label = self.labels.index(text_label)
        target = torch.zeros(len(self.labels))
        target[label] = 1
        
        # print("text_label", text_label)
        # print("label", label)
        
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        return audio,target



class SoundscapeDataset(Dataset):
    def __init__(self, n_samples=10000,  soundscape_dir="random_soundscapes"):
    
        self.files = [f for f in listdir(soundscape_dir) if (f[-4:] == ".wav")]






    def __len__(self):
        return len(self.metadata_file)*self.n_per_file

    def __getitem__(self, idx):

        which = math.floor(idx/(len(self.metadata_file)))
  



class MusicDataset(Dataset):
    def __init__(self, n_samples=32000, n_per_file = 100, audio_dir="../sounds/songsinmyhead_2016", transform=None, target_transform=None):
        

        self.files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]
        self.files = self.files[:20]

        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

        self.n_samples = n_samples
        self.n_per_file = n_per_file
        self.labels = self.files


    def __len__(self):
        return len(self.labels)*self.n_per_file

    def __getitem__(self, idx):

        which = math.floor(idx/(len(self.files)))
  
        audio_path = os.path.join(self.audio_dir, self.files[idx % len(self.files)])
        audio, rate = torchaudio.load(audio_path)
        audio = audio[0] # use only left channel

        
        audio = audio[44100 + which*self.n_samples: 44100 + (which + 1)*self.n_samples] 
        audio = audio.reshape([1, -1]) # add channel field back

        
        label = idx % len(self.files)
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        return audio, label



if __name__ == "__main__":
    #full_dataset = BirdClefDataset(target_transform=Lambda(lambda y: torch.zeros(21, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    full_dataset = BirdClefDataset()
    print('labels', full_dataset.labels)
    print('len labels', len(full_dataset.labels))

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    training_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
    # Display image and label.
    start_time = timer()
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print("labels shape", train_labels.size())
    print('train labels', train_labels)
    print('train labels 0', train_labels[0])
    end_time = timer()
    print("time", end_time - start_time)

