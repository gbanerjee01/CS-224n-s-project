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
  def __init__(self, df, input_len):
    self.df = df.reset_index(drop=True)
    self.length=input_len

    self.max_len = 0
    padded = []
    for idx in range(len(self.df)):
      m = self.df.loc[idx, 'mel']
      self.max_len = max(self.max_len, m.shape[1])
      if m.shape[1] >= input_len:
        m = m[:input_len]
      else:
        temp = np.zeros((m.shape[0], input_len))
        temp[:,:m.shape[1]] = m
        m = temp
      padded.append(m)
    self.df['mel_pad'] = padded
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    output_data = {}
    sample = {
        'mel': self.df.loc[idx, 'mel_pad'],
        'mfcc': self.df.loc[idx, 'mfcc_pad'],
        'chromagram': self.df.loc[idx, 'chromagram_pad'],
        'emotion': self.df.loc[idx, 'emotion'],
        'intensity': self.df.loc[idx, 'intensity'],
        'statement': self.df.loc[idx, 'statement'],
        'repeat': self.df.loc[idx, 'repeat'],
        'gender': self.df.loc[idx, 'gender']
    }

    output_data = {}
    values = sample["mel"].reshape(-1, 256, self.length)
    values = torch.Tensor(values)

    target = torch.LongTensor([sample["emotion"]])

    return (values, target)

def fetch_dataloader(df, batch_size, num_workers):
    dataset = RavdessDataset(df, 250)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader