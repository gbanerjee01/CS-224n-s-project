from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import pickle
import torch

class TransformerDataset(Dataset):
  def __init__(self, df, features):
    self.labels = df['emotion'].reset_index(drop=True)
    self.features = {}
    for feat in features:
        self.features[feat] = df[feat + '_pad'].reset_index(drop=True)
    self.image_size = 384
    self.transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((self.image_size, self.image_size))])
                                          #transforms.Normalize([0.5, 0.5, 0.5],
                                                              #[0.5, 0.5, 0.5]),])
      
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    data = np.concatenate([self.features[feat].iloc[idx] for feat in self.features], axis=0)
    # pad 1st dimension to the 0th dimension
    padded_data = np.zeros((data.shape[0], data.shape[0]))
    padded_data[:,:data.shape[1]] = data
    img = self.transform(padded_data)
    # stack the same data three times to emulate RGB image
    img = torch.stack([img]*3, dim=1).squeeze(dim=0).type(torch.float)

    label = self.labels.iloc[idx]        
    label = torch.tensor(label-1).type(torch.long)
    
    return img, label

def fetch_dataloader(df, batch_size, num_workers, features):
    dataset = TransformerDataset(df, features)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader