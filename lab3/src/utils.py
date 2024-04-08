import torch
import torch.nn as nn
import numpy as np

class DICEloss(nn.Module):
    def __init__(self) -> None:
        super(DICEloss, self).__init__()
        self.eps = 1e-5
        
    def forward(self, pred, target):
        intersection = 2 * torch.sum(pred * target) + self.eps
        union = torch.sum(pred) + torch.sum(target) + self.eps
        loss = 1 - intersection / union
        return loss
    

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(3, 1, 100, 100)
    y = torch.randn(3, 1, 100, 100)
    
    criterion = DICEloss()
    score = criterion(x, y)
    print(score)