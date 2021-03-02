from torch.utils.data import BatchSampler, Dataset, DataLoader
import lmdb
import torchvision
import pandas as pd
import numpy as np
import pickle
import torch
from PIL import Image
import torch
from torch.utils.data import Dataset

#edited this whole file

class RavdessDataset(Dataset):
  def __init__(self, df):
    self.df = df.reset_index(drop=True)
    self.length=250
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    output_data = {}
    sample = {
        'M': self.df.loc[idx, 'M'],
        'mfcc': self.df.loc[idx, 'mfcc'],
        'chromagram': self.df.loc[idx, 'chromagram'],
        'emotion': self.df.loc[idx, 'emotion'],
        'intensity': self.df.loc[idx, 'intensity'],
        'statement': self.df.loc[idx, 'statement'],
        'repeat': self.df.loc[idx, 'repeat'],
        'gender': self.df.loc[idx, 'gender']
    }

    output_data = {}
    values = sample["M"].reshape(-1, 128, self.length)
    values = torch.Tensor(values)

    target = torch.LongTensor([sample["emotion"]])

    return (values, target)

def fetch_dataloader(df, batch_size, num_workers):
    dataset = AudioDataset(df)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader