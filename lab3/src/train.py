import os
import math
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import DICEloss
from evaluate import evaluate
from models.unet import UNet
from models.resnet34_unet import resnet34_unet
from oxford_pet import load_dataset


def train(cfg, model, train_set, valid_set):
    # initialize
    epoch_init = 1
    best_val = 0.0
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []
    
    os.makedirs('./saved_models/weight', exist_ok=True)
    os.makedirs('./saved_models/checkpoint', exist_ok=True)
    
    # dataset, dataloader
    train_dataset, train_dataloader = train_set
    valid_dataset, valid_dataloader = valid_set
    
    # criterion
    criterion = DICEloss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # use checkpoint
    if cfg.ckpt_use == True:
        try:
            ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if cfg.history == True:
                epoch_init = ckpt['epoch'] + 1
                train_epoch_loss = ckpt['train_loss']
                train_epoch_acc = ckpt['train_acc']
                val_epoch_loss = ckpt['valid_loss']
                val_epoch_acc = ckpt['valid_acc']
            print(f'> Using {cfg.ckpt_path}')
            print(f'> {"Keep" if cfg.history else "Clean"} history -> start training from {epoch_init} epoch')
        except Exception as e:
            print(e)
    else:
        print(f'> No using checkpoint')
    
    # training loop
    train_start = time.time()
    for epoch in range(epoch_init, cfg.epochs + 1):
        # train
        epoch_start = time.time()
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for sample in tqdm(train_dataloader):
            # train
            x, y = sample['image'].to(cfg.device), sample['mask'].to(cfg.device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            score = 1 - loss
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # performance
            total_loss += loss.cpu().detach().numpy()
            total_acc += score.cpu().detach().numpy()
        
        training_loss = total_loss / math.ceil(len(train_dataset) / cfg.batch_size)
        training_acc = total_acc / math.ceil(len(train_dataset) / cfg.batch_size)
        
        train_epoch_loss.append(training_loss)
        train_epoch_acc.append(training_acc)
        
        # validate
        valid_loss, valid_acc = evaluate(model, cfg, valid_dataset, valid_dataloader, criterion)
        val_epoch_loss.append(valid_loss.cpu().detach().numpy())
        val_epoch_acc.append(valid_acc.cpu().detach().numpy())
        
        print('Epoch {}/{} > training loss: {:.4f}, training acc: {:.4f}, valid loss: {:.4f}, valid acc: {:.4f}, duration: {:.4f}'
                .format(epoch, cfg.epochs, training_loss, training_acc, valid_loss, valid_acc, time.time() - epoch_start))
        
        if valid_acc > best_val:
            best_val = valid_acc
            if cfg.data_augmentation:
                weight_path = f'./saved_models/weight/best_{cfg.model}_da.pth'
            else:
                weight_path = f'./saved_models/weight/best_{cfg.model}.pth'
            torch.save(model.state_dict(), weight_path)
            
            print('> Save the best model weight in epoch {} - val_acc: {:.4f}'.format(epoch, best_val))
        
        if epoch % cfg.save_weight == 0:
            if cfg.data_augmentation:
                weight_path = f'./saved_models/weight/{cfg.model}_{epoch}_da.pth'
                plot_path = f'./{cfg.model}_epoch{epoch}_acc_da.png'
            else:
                weight_path = f'./saved_models/weight/{cfg.model}_{epoch}.pth'
                plot_path = f'./{cfg.model}_epoch{epoch}_acc.png'
            
            torch.save(model.state_dict(), weight_path)
            print(f'> Save the model weight at {weight_path}')
            
            plt.title(f'{cfg.model} performance'), plt.ylabel('accuracy'), plt.xlabel('epoch')
            plt.plot(range(1, epoch + 1), train_epoch_acc, 'b', label='Training acc')
            plt.plot(range(1, epoch + 1), val_epoch_acc, 'r', label='Validation acc')
            plt.legend(loc='lower right')
            plt.savefig(plot_path)
            plt.clf()
            print(f'> Save the plot of the accuracy in {plot_path}')          
        
        if epoch % cfg.save_ckpt == 0:
            if cfg.data_augmentation:
                path = f'./saved_models/checkpoint/{cfg.model}_epoch{epoch}_da.ckpt'
            else:
                path = f'./saved_models/checkpoint/{cfg.model}_epoch{epoch}.ckpt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'train_acc': train_epoch_acc,
                'valid_loss': val_epoch_loss,
                'valid_acc': val_epoch_acc,
            }, path)
            print(f'> Save the checkpoint in {path}')
            
    print('Total training time: {:.4f}'.format(time.time() - train_start))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    # model
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='choice the model')
    
    # dataset
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--data_augmentation', '-da', action='store_true', help='use data augmentation or not')
    
    # training
    parser.add_argument('--ckpt_use', '-cu', action='store_true', help='use checkpoint for training or not')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--history', action='store_true', help='keep loss history')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', '-nw', type=int, default=0 ,help='number of workers')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='device for training')
    
    parser.add_argument('--save_weight', type=int, default=50, help='epoch interval of saving model weight')
    parser.add_argument('--save_ckpt', type=int, default=20, help='epoch interval of saving checkpoint')
    
    return parser.parse_args()


if __name__ == "__main__":
    # hyperparameters
    args = get_args()
    print('> Hyperparamer')
    print('> model - {}, epochs - {}, lr - {}, batch_size - {}, num_workers - {}, device - {}'
            .format(args.model, args.epochs, args.lr, args.batch_size, args.num_workers, args.device))
    print('> save_weight - {}, save_ckpt - {}'.format(args.save_weight, args.save_ckpt))
    print('----------------------------------------')
    
    # model
    model = UNet() if args.model == 'unet' else resnet34_unet()
    model = model.to(device=args.device)
    
    # dataset
    train_dataset = load_dataset(args.data_path, 'train', data_augentation=args.data_augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    valid_dataset = load_dataset(args.data_path, 'valid', data_augentation=args.data_augmentation)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    train_set = (train_dataset, train_dataloader)
    valid_set = (valid_dataset, valid_dataloader)
    
    train(args, model, train_set, valid_set)