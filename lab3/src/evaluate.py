import math
import torch

def evaluate(model, cfg, valid_dataset, valid_dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for sample in valid_dataloader:
            x, y = sample['image'].to(cfg.device), sample['mask'].to(cfg.device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            score = 1 - loss

            total_loss += loss
            total_acc += score

    valid_loss = total_loss / math.ceil(len(valid_dataset) / cfg.batch_size)
    valid_acc = total_acc / math.ceil(len(valid_dataset) / cfg.batch_size)
    
    return valid_loss, valid_acc