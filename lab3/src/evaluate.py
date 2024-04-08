import torch

def evaluate(model, cfg, valid_dataset, valid_dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for sample in valid_dataloader:
            x, y = sample['image'].to(cfg.device), sample['mask'].to(cfg.device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            total_loss += loss
            total_acc += torch.sum(((y_pred > 0.5) * 1) * y) / torch.sum(y)

    valid_loss = total_loss / len(valid_dataset)
    valid_acc = total_acc / len(valid_dataset)
    
    return valid_loss, valid_acc