# Diffusion model
import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import tqdm_bar, plot_curve
from data import ICLEVR_Dataset
from evaluator import evaluation_model
from model.ddpm import DDPM
from model.unet import Unet


CUDA_VISIBLE_DEVICES='1'


def main(args):
    # params.
    device = args.device
    ckpt_save_path = args.output_ckpt +'/' + 'dm'
    output_path = args.output + '/' + 'dm'
    os.makedirs(ckpt_save_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # dataset
    train_dataset = ICLEVR_Dataset(args.root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    test_dataset = ICLEVR_Dataset(args.root, test_file=args.test_file, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    new_test_dataset = ICLEVR_Dataset(args.root, test_file=args.new_test_file, mode='test')
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    # model
    unet_model = Unet().to(args.device)
    unet_model.initialize_weights()
    
    ddpm = DDPM(unet_model=unet_model, betas=(args.beta_start, args.beta_end), noise_steps=args.step, device=args.device).to(device)
    
    # loss
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=5, min_lr=0, verbose=True)
    evaluator = evaluation_model(device)
    
    # training loop
    train(args, device, ddpm, optimizer, scheduler, evaluator, 
          train_dataloader, test_dataloader, new_test_dataloader, ckpt_save_path, output_path)


def train(  args, device,
            ddpm, optim, scheduler, evaluator, 
            train_dataloader, test_dataloader, new_test_dataloader, 
            ckpt_save_path, output_path):
    best_acc, new_best_acc = 0.0, 0.0
    acc_list, new_acc_list = [], []
    loss_total = []
    
    for epoch in range(1, args.epochs+1):
        ddpm.train()
        optim.param_groups[0]['lr'] = args.lr*(1-epoch/args.epochs)
        loss_ema = None
        
        for x, cond in (pbar := tqdm(train_dataloader)):
            x, cond = x.to(device), cond.to(device)
            
            loss = ddpm(x, cond)
            if loss_ema is None:
                    loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            desp = f'Train - Epoch {epoch}/{args.epochs} - lr: {scheduler.get_last_lr()[0]}, Loss: {loss_ema:.4f}'
            tqdm_bar(pbar, desp)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        loss_total.append(loss_ema)
        
        if epoch % args.ckpt_save == 0:
            path = os.path.join(output_path, f'Epoch_{epoch}')
            os.makedirs(path, exist_ok=True)
            
            # plot loss curve
            avg_loss = sum(loss_total)/len(loss_total)
            title, curve = f'Train_Epoch_{epoch}_MSELoss', f'DDPM - avg. loss={avg_loss:.3f}'
            plot_curve( title, len(loss_total), loss_total, curve, path=os.path.join(path, 'Loss.png'))
            
            # validation - test file
            acc, grid = valid(ddpm, evaluator, test_dataloader, device)
            acc_list.append(acc)
            save_image(grid, os.path.join(path, 'test.png'))
            if acc > best_acc:
                best_acc = acc
                torch.save(ddpm.state_dict(), f'{output_path}/ddpm_best.pth')
                print(f'>>> Best Acc: {best_acc}')
                print(f'> Save weight at {output_path}/ddpm_best.pth')
            
            # validation - new test file
            acc, grid = valid(ddpm, evaluator, new_test_dataloader, device)
            new_acc_list.append(acc)
            save_image(grid, os.path.join(path, 'new_test.png'))
            if acc > new_best_acc:
                new_best_acc = acc
                torch.save(ddpm.state_dict(), f'{output_path}/ddpm_new_best.pth')
                print(f'>>> New Best Acc: {new_best_acc}')
                print(f'> Save weight at {output_path}/ddpm_new_best.pth')

            # plot accuracy curve
            title, curve1, curve2 = f'Train_Epoch_{epoch}_Accuracy', f'test set(Best={best_acc:.3f})', f'new test set(Best={new_best_acc:.3f})'
            plot_curve( title, len(acc_list), acc_list, curve1, new_acc_list, curve2, path=os.path.join(path, 'Accuracy.png'))
            
            # save checkpoint
            path = os.path.join(ckpt_save_path, f'ddpm_Epoch_{epoch}.ckpt')
            torch.save({
                "ddpm":         ddpm.state_dict(),
                "optim":        optim.state_dict(),
                "scheduler":    scheduler.state_dict()
            }, path)
            print(f'> Save checkpoint at {path}')
        
        if epoch % args.pt_save == 0:
            path = os.path.join(output_path, f'Epoch_{epoch}')
            torch.save(ddpm.state_dict(), f'{path}/ddpm_Epoch_{epoch}.pth')
            print(f'> Save weight at', os.path.join(path, f'ddpm_Epoch_{epoch}.pth'))
        
        scheduler.step(best_acc+new_best_acc)


def valid(ddpm, evaluator, test_dataloader, device):
        ddpm.eval()
        x_gen, label = [], []
        with torch.no_grad():
            for cond in tqdm(test_dataloader):
                cond = cond.to(device)
                x_i = ddpm.sample(cond, (3, 64, 64), device)
                x_gen.append(x_i)
                label.append(cond)
            x_gen, label = torch.stack(x_gen, dim=0).squeeze(), torch.stack(label, dim=0).squeeze()
            
            acc = evaluator.eval(x_gen, label)
            grid = make_grid(x_gen, nrow=8, normalize=True)
            print(f'Valid - Accuracy: {acc:.4f}')
            
        return acc, grid
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    
    # dataset
    parser.add_argument('--root', type=str, default='./dataset', help='training data directory')
    parser.add_argument('--test_file', type=str, default='test.json', help='testing data file')
    parser.add_argument('--new_test_file', type=str, default='new_test.json', help='new testing data file')
    
    # model
    parser.add_argument('--step', type=int, default=300, help='timestep to process')
    parser.add_argument('--beta_start', type=int, default=1e-4, help='start beta value')
    parser.add_argument('--beta_end', type=int, default=0.02, help='end beta value')
    
    # training hyperparam.
    parser.add_argument('--device', type=str, default='cuda:1', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=8, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size of data')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='maximum epoch to train')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--use_ckpt', action='store_true', help='use checkpoint for training')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/epoch=20.ckpt')
    
    # saving param.
    parser.add_argument('--pt_save', type=int, default=10, help='model weight saving interval')
    parser.add_argument('--ckpt_save', type=int, default=5, help='checkpoint saving interval')
    parser.add_argument('--output', type=str, default='./output', help='training results saving path')
    parser.add_argument('--output_ckpt', type=str, default='./checkpoint', help='training results saving path')
    
    args = parser.parse_args()
    
    main(args)