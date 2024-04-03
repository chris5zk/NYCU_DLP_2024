import warnings
warnings.simplefilter("ignore", UserWarning)

import time
import torch
import argparse
from tqdm import tqdm
from VGG19 import vgg19
from ResNet50 import resnet50
from dataloader import DataHandler


def evaluate():
    print("evaluate() not defined")

def test():
    print("test() not defined")

def train(model, datahandler, cfg):
    # initialize
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []
    
    # train set
    train_dataset = datahandler.get_dataset('train')
    train_dataloader = datahandler.get_dataloader(train_dataset, cfg.batch_size, cfg.num_workers)
    
    # valid set
    valid_dataset = datahandler.get_dataset('valid')
    valid_dataloader = datahandler.get_dataloader(valid_dataset, cfg.batch_size, cfg.num_workers)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # training loop
    train_start = time.time()
    for epoch in range(cfg.epoch_nums):    
        epoch_start = time.time()
        
        model.train()
        total_loss, total_acc = 0.0, 0.0   
        for x, y in tqdm(train_dataloader):
            # to device
            x, y = x.to(cfg.device), y.to(cfg.device)
            
            # predict
            y_pred = model(x)
            
            # loss
            loss = criterion(y_pred, y)
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # performance
            total_loss += loss
            total_acc += (torch.max(y_pred, 1)[1] == y).sum().item()
        
        training_loss = total_loss / len(train_dataset)
        training_acc = total_acc / len(train_dataset)
        
        train_epoch_loss.append(training_loss)
        train_epoch_acc.append(training_acc)
        
        print('Epoch {} > training loss: {:.4f}, training acc: {:.4f}, duration: {:.4f}'.format(epoch+1, training_loss, training_acc, time.time() - epoch_start))

        if (epoch + 1) % cfg.val_interval == 0:   
            val_start = time.time()
    
            model.eval()
            total_loss, total_acc = 0.0, 0.0
            with torch.no_grad():
                for x, y in tqdm(valid_dataloader):
                    # to device
                    x, y = x.to(cfg.device), y.to(cfg.device)
                    
                    # predict
                    y_pred = model(x)
                    
                    # loss
                    loss = criterion(y_pred, y)
                    
                    # performance
                    total_loss += loss
                    total_acc += (torch.max(y_pred, 1)[1] == y).sum().item()

            valid_loss = total_loss / len(valid_dataset)
            valid_acc = total_acc / len(valid_dataset)
            
            val_epoch_loss.append(valid_loss)
            val_epoch_acc.append(valid_acc)
            
            print('Validate > val loss: {:.4f}, val acc: {:.4f}, duration: {:.4f}'.format(epoch+1, valid_loss, valid_acc, time.time() - val_start))
    
        if (epoch + 1) % cfg.save_weight == 0:
            torch.save(model.state_dict(), f'./weight/{cfg.model}_{epoch+1}.pt')
        
        if (epoch + 1) % cfg.save_ckpt == 0:
            path = f'./checkpoint/{cfg.model}_epoch{epoch+1}.ckpt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'train_acc': train_epoch_acc,
                'valid_loss': val_epoch_loss,
                'valid_acc': val_epoch_acc,
            }, path)
    
    print('Total training time: {:.4f}'.format(time.time() - train_start))

def main(cfg):
    # datahandler
    datahandler = DataHandler(cfg.root, cfg.dataset)
    
    # model setting
    model = vgg19() if cfg.model == 'vgg19' else resnet50()
    model = model.to(device=cfg.device)
    
    # train model
    train(model=model, datahandler=datahandler, cfg=cfg)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate'], help='model mode: "train", "test"')
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg19', 'resnet50'], help='model type: "vgg19", "resnet50"')
    parser.add_argument('--pretrained', type=str, default='./weight/best.pt', help='load pre-trained model path')
    
    # dataset
    parser.add_argument('--root', type=str, default='./dataset', help='the root of dataset')
    parser.add_argument('--dataset', type=str, default='ButterflyMoth', help='dataset name')
    
    # training
    parser.add_argument('--ckpt', type=str, default='./checkpoint/best.ckpt', help='checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--epoch_nums', type=int, default=500, help='max number of epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='numbers of workers')
    parser.add_argument('--save_weight', type=int, default=50, help='epoch interval of saving model weight')
    parser.add_argument('--save_ckpt', type=int, default=20, help='epoch interval of saving checkpoint')
    parser.add_argument('--val_interval', type=int, default=5, help='epoch interval of running validation set')
    
    args = parser.parse_args()

    main(args)