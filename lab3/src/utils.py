import torch
import torch.nn as nn
import numpy as np

class DICEloss(nn.Module):
    def __init__(self) -> None:
        super(DICEloss, self).__init__()
        self.eps = 1e-5
        
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + self.eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.eps)))
        
        return loss.mean()
    

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(3, 1, 100, 100)
    y = torch.randn(3, 1, 100, 100)
    
    criterion = DICEloss()
    score = criterion(x, y)
    print(score)