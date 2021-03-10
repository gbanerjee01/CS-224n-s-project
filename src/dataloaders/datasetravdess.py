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
    
    sample = {
        'M': self.df.loc[idx, 'mel_pad'],
        'mfcc': self.df.loc[idx, 'mfcc_pad'],
        'chromagram': self.df.loc[idx, 'chromagram_pad'],
        'spec_contrast': self.df.loc[idx, 'spec_contrast_pad'],
        'tonnetz': self.df.loc[idx, 'tonnetz_pad'],
        'emotion': self.df.loc[idx, 'emotion'],
        # 'intensity': self.df.loc[idx, 'intensity'],
        # 'statement': self.df.loc[idx, 'statement'],
        # 'repeat': self.df.loc[idx, 'repeat'],
        # 'gender': self.df.loc[idx, 'gender']
    }

    # values = np.concatenate([sample[k].reshape(-1) for k in ['M', 'mfcc', 'chromagram', 'spec_contrast', 'tonnetz']])
    values = np.concatenate([sample[k] for k in ['M', 'mfcc', 'chromagram', 'spec_contrast', 'tonnetz']])
    # values = np.expand_dims(values, axis=1)
    # values = values.transpose(1,0)
    values = torch.Tensor(values)

    target = torch.LongTensor([sample["emotion"]])

    return (values, target)

def fetch_dataloader(df, batch_size, num_workers):
    dataset = RavdessDataset(df, 250)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader