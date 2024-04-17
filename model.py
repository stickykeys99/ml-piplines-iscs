import torch
from torch import nn
from torch.utils.data import Dataset

class Model(nn.Module):
    def __init__(self, features, min_x, max_x):
        super().__init__()
        self.features = features
        self.min_x = min_x
        self.max_x = max_x

        self.layers = nn.Sequential(
            nn.Linear(features, 36),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(36, 16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(4,1)
        )
    
    def forward(self, x):
        x = (x - self.min_x) / (self.max_x - self.min_x)
        return self.layers(x)
    
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.n_samples = len(x)
        self.features = 0 if len(x.shape) < 2 else x.shape[-1]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples