from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import pickle
import torch

class TransformerDataset(Dataset):
  def __init__(self, df):
    self.labels = df['emotion'].reset_index(drop=True)
    self.num_labels = self.labels.nunique()
    self.mels = df['mel_pad'].reset_index(drop=True)
    self.max_len = self.mels[0].shape[1]
    self.image_size = 384
    self.transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((self.image_size, self.image_size))])
                                          #transforms.Normalize([0.5, 0.5, 0.5],
                                                              #[0.5, 0.5, 0.5]),])
      
  def __len__(self):
    return len(self.mels)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
        
    mel = self.mels.iloc[idx]
    label = self.labels.iloc[idx]        
    mel = self.transform(mel[:250,])
    # stack the same mel spectrogram three times to emulate RGB image
    mel = torch.stack([mel]*3, dim=1).squeeze(dim=0).type(torch.float)
    label = torch.tensor(label-1).type(torch.long)
    
    return mel, label

def fetch_dataloader(df, batch_size, num_workers):
    dataset = TransformerDataset(df)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader