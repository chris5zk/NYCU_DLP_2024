import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from models import MaskGit as VQGANTransformer
from utils import LoadTrainData


def tqdm_bar(pbar, desp, loss):
    pbar.set_description(desp , refresh=False)
    pbar.set_postfix(loss=float(loss), refresh=False)
    pbar.refresh()

def plot(title, epoch, y1, y2, curve1, curve2, path):
    x = range(0, epoch)
    plt.title(title)
    plt.plot(x, y1, 'b', label=curve1)
    plt.plot(x, y2, 'r', label=curve2)
    plt.legend(loc='lower right')
    plt.savefig(path)
    plt.clf()

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers(args.lr)
        self.prepare_training(args)
        
    def prepare_training(self, args):
        self.pt_path, self.ckpt_path, self.plot_path = './output/transformer_weights', './output/transformer_checkpoints', './output/plots'
        os.makedirs(self.pt_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)

    def train_one_epoch(self, args, epoch, train_loader, criterion):
        epoch_loss = 0.0
        for x in (pbar := tqdm(train_loader)):
            x = x.to(args.device)
            
            # predict
            logits, target = self.model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            # optimize
            self.optim.zero_grad()
            loss.backward()
            self.optim.step
            
            # tqdm bar
            desp = f'Train - Epoch {epoch}/{args.epochs}, lr:{self.optim.state_dict()["param_groups"][0]["lr"]}'
            tqdm_bar(pbar, desp, loss)
            
            epoch_loss += loss
            
        epoch_loss /= len(train_loader)
            
        if epoch % args.pt_save_per_epoch == 0:
            torch.save(self.model.state_dict(), f'./transformer_weights/epoch={epoch}.pt')
            print(f'> Save model weight at ./transformer_weights/epoch={epoch}.pt')
            
        if epoch % args.ckpt_save_per_epoch == 0:
            torch.save({
                "state_dict":   self.model.state_dict(),
                "optimizer" :   self.optim.state_dict(),
                "lr"        :   self.scheduler.state_dict(),
                "last_epoch":   epoch
            }, f'./transformer_checkpoints/epoch={epoch}.ckpt')
            print(f'> Save checkpoint at ./transformer_checkpoints/epoch={epoch}.ckpt')
        
        self.scheduler.step(epoch_loss)
        
        return epoch_loss

    def eval_one_epoch(self, args, val_loader, criterion):
        epoch_loss = 0.0
        for x in (pbar := tqdm(val_loader)):
            x = x.to(args.device)
            
            # predict
            logits, target = self.model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            # loss
            epoch_loss += loss
        
        epoch_loss /= len(val_loader)
        
        return epoch_loss

    def configure_optimizers(self, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.96))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer, scheduler


def main(args):
    # model config
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    # dataset
    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    # loss
    criterion = nn.CrossEntropyLoss()
    
    # checkpoint
    ###
    
    
#TODO2 step1-5:
    train_loss_total, val_loss_total = [], []
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        # training stage
        train_loss = train_transformer.train_one_epoch(args, epoch, train_loader, criterion)
        train_loss_total.append(train_loss)
        avg_train_loss = sum(train_loss_total)/len(train_loss_total)
        
        # validate stage
        val_loss = train_transformer.eval_one_epoch(args, epoch, val_loader, criterion)
        val_loss_total.append(val_loss)
        avg_val_loss = sum(val_loss_total)/len(val_loss_total)

        print(f'> Training Loss - {train_loss:.3f}, Training Avg. - {avg_train_loss:.3f}, Validation Loss - {val_loss:.3f}, Validation Avg. {avg_val_loss:.3f}') 

        if epoch % args.pt_save_per_epoch == 0:
            path = './output/plots' + '/' + f'epoch={epoch}.png'
            title = f'Epoch {epoch} performance'
            curve1 = f'Training Loss: avg. {avg_train_loss:.2f}'
            curve2 = f'Validate Loss: avg. {avg_val_loss:.2f}'
            plot(title, train_loss_total, val_loss_total, curve1, curve2, path)
            print(f'> Save loss curve at {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Dataset splitten')
    parser.add_argument('--accum_grad', type=int, default=10, help='Number for gradient accumulation.')

    # you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--pt_save_per_epoch', type=int, default=1, help='Save PT per ** epochs(defcault: 10)')
    parser.add_argument('--ckpt_save_per_epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    args = parser.parse_args()
    
    main(args)