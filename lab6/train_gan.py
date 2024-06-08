# DCGAN
import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import tqdm_bar, plot_curve
from data import ICLEVR_Dataset
from evaluator import evaluation_model
from model.gan import Generator, Discriminator

GPUS='0,1'
CUDA_VISIBLE_DEVICES=GPUS
ngpu=2


def main(args):
    # params.
    device = args.device
    ckpt_save_path = f'./{args.output_ckpt}/gan'
    output_path = f'./{args.output}/gan'
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
    netG = Generator(nz=args.Z_dims, ngf=args.G_dims).to(device)
    netD = Discriminator(ndf=args.D_dims).to(device)
    
    # checkpoints
    if args.use_ckpt:
        netG.load_state_dict(torch.load(args.ckpt_path))
        netD.load_state_dict(torch.load(args.ckpt_path))
    
    # loss
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    evaluator = evaluation_model()
    
    # training loop
    train(args, netG, netD, optimizerG, optimizerD, criterion, evaluator, train_dataloader, test_dataloader, new_test_dataloader, device, ckpt_save_path, output_path)
    

def train(args, netG, netD, optimG, optimD, criterion, evaluator, train_dataloader, test_dataloader, new_test_dataloader, device, ckpt_save_path, output_path):
    best_acc, new_best_acc = 0.0, 0.0
    acc_list, new_acc_list = [], []
    loss_d_total, loss_g_total = [], []
    
    for epoch in range(1, args.epochs+1):
        netD.train()
        netG.train()
        loss_d_epoch, loss_g_epoch = 0.0, 0.0
        
        for x, y in (pbar := tqdm(train_dataloader)):
            batch_size = x.size(0)
            real_x = x.to(device)
            condition = y.to(device)
            
            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_y = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)
            fake_y = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)
            if random.random() < 0.1:
                real_y, fake_y = fake_y, real_y
                
            # train with real
            output = netD(real_x, condition)
            D_x = output.mean().item()
            loss_D_real = criterion(output, real_y)
            
            # train with fake
            noise = torch.randn(batch_size, args.Z_dims, 1, 1, device=device)
            fake_x = netG(noise, condition)
            output = netD(fake_x.detach(), condition)
            D_G_z1 = output.mean().item()
            loss_D_fake = criterion(output, fake_y)
            
            # total loss
            loss_D = loss_D_real + loss_D_fake

            optimD.zero_grad()
            loss_D.backward()
            optimD.step()
            
            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ########################### 
            generator_label = torch.ones(batch_size).to(device)  # fake labels are real for generator cost
            output = netD(fake_x.detach(), condition)
            D_G_z2 = output.mean().item()
            
            loss_G = criterion(output, generator_label)

            optimG.zero_grad()
            loss_G.backward()
            optimG.step()
            
            loss_d_epoch += loss_D.item()
            loss_g_epoch += loss_G.item()
            
            desp = f'Train - Epoch {epoch}/{args.epochs}, Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}'
            tqdm_bar(pbar, desp)
        
        loss_d_epoch /= len(train_dataloader)
        loss_g_epoch /= len(train_dataloader)
        loss_d_total.append(loss_d_epoch)
        loss_g_total.append(loss_g_epoch)
        
        if epoch % args.ckpt_save == 0:
            output_iter_path = output_path + '/' + f'Epoch_{epoch}'
            os.makedirs(output_iter_path, exist_ok=True)
            # print performance
            avg_loss_G = sum(loss_g_total)/len(loss_g_total)
            avg_loss_D = sum(loss_d_total)/len(loss_d_total)
            print('Train - Epoch %d/%d: Loss_G: %.4f(avg. %.4f), Loss_D: %.4f(avg. %.4f), D(x): %.4f, D(G(z)): %.4f / %.4f'
                    % (epoch, args.epochs, loss_g_epoch, avg_loss_G, loss_d_epoch, avg_loss_D, D_x, D_G_z1, D_G_z2))
            
            # save loss plot
            title, curve1, curve2 = f'Train_Epoch_{epoch}_BCELoss', f'netD - avg. loss: {avg_loss_D:.3f}', f'netG - avg. loss: {avg_loss_G:.3f}'
            file_path = output_iter_path + '/' + 'Loss.png'
            plot_curve(title, epoch, loss_d_total, loss_g_total, curve1, curve2, file_path)
            
            # validation & save gen. images
            acc, new_acc = valid(args, netG, netD, evaluator, test_dataloader, new_test_dataloader, best_acc, new_best_acc, device, output_path, output_iter_path)
            acc_list.append(acc)
            new_acc_list.append(new_acc)
            
            # save acc. plot
            title, curve1, curve2 = f'Train_Epoch_{epoch}_Acc', 'test set', 'new test set'
            file_path = output_iter_path + '/' + 'Accuracy.png'
            plot_curve(title, epoch, acc_list, new_acc_list, curve1, curve2, file_path, loc='lower right')
            
            
            if acc > best_acc:
                best_acc = acc
                torch.save(netG.state_dict(), f'{output_path}/netG_best.pth')
                torch.save(netD.state_dict(), f'{output_path}/netD_best.pth')
                print(f'Best Acc: {best_acc}')
                print(f'> Save weight at {output_path}/netG_best.pth')
                print(f'> Save weight at {output_path}/netD_best.pth')
            
            if new_acc > new_best_acc:
                new_best_acc = acc
                torch.save(netG.state_dict(), f'{output_path}/netG_new_best.pth')
                torch.save(netD.state_dict(), f'{output_path}/netD_new_best.pth')
                print(f'New Best Acc: {new_best_acc}')
                print(f'> Save weight at {output_path}/netG_new_best.pth')
                print(f'> Save weight at {output_path}/netD_new_best.pth')
            
            # save checkpoint
            save_G_model = ckpt_save_path + '/' + f'netG_Epoch_{epoch}.ckpt'
            torch.save({
                "netG":     netG.state_dict(),
                "optimG" :  optimG.state_dict(),
            }, save_G_model)
            print(f'> Save checkpoint at {save_G_model}')
            
            save_D_model = ckpt_save_path + '/' + f'netD_Epoch_{epoch}.ckpt'
            torch.save({
                "netD":     netD.state_dict(),
                "optimD" :  optimD.state_dict(),
            }, save_D_model)
            print(f'> Save checkpoint at {save_D_model}')
            
        if epoch % args.pt_save == 0:
            # save weights
            output_iter_path = output_path + '/' + f'Epoch_{epoch}'
            torch.save(netG.state_dict(), f'{output_iter_path}/netG_Epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'{output_iter_path}/netD_Epoch_{epoch}.pth')
            print(f'> Save weight at {output_iter_path}/netG_Epoch_{epoch}.pth')
            print(f'> Save weight at {output_iter_path}/netD_Epoch_{epoch}.pth')
                

def valid(args, netG, netD, evaluator, test_dataloader, new_test_dataloader, best_acc, new_best_acc, device, output_path, output_iter_path):
    netG.eval()
    netD.eval()
    with torch.no_grad():
        for y in test_dataloader:
            condition = y.to(device)
            batch_size = y.shape[0]
            
            noise = torch.randn(batch_size, args.Z_dims, 1, 1, device=device)
            gen_image = netG(noise, condition)
            save_image(gen_image.detach(), output_iter_path + '/' + f'test.png')
            
            acc = evaluator.eval(gen_image, condition)
            print(f'Valid: test - Accuracy: {acc:.4f}')
                
        for y in new_test_dataloader:
            condition = y.to(device)
            batch_size = y.shape[0]
            
            noise = torch.randn(batch_size, args.Z_dims, 1, 1, device=device)
            gen_image = netG(noise, condition)
            save_image(gen_image.detach(), output_iter_path + '/' + f'new_test.png')
            
            new_acc = evaluator.eval(gen_image, condition)
            print(f'Valid: new test - Accuracy: {new_acc:.4f}')

    return acc, new_acc
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    
    # dataset
    parser.add_argument('--root', type=str, default='./dataset', help='training data directory')
    parser.add_argument('--test_file', type=str, default='test.json', help='testing data file')
    parser.add_argument('--new_test_file', type=str, default='new_test.json', help='new testing data file')
    
    # model
    parser.add_argument('--Z_dims', type=int, default=100, help='latent dimension')
    parser.add_argument('--G_dims', type=int, default=256, help='generator feature dimension')
    parser.add_argument('--D_dims', type=int, default=256, help='discriminator feature dimension')
    
    # training hyperparam.
    parser.add_argument('--device', type=str, default='cuda:0', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=8, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size of data')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='maximum epoch to train')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--use_ckpt', action='store_true', help='use checkpoint for training')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/epoch=20.ckpt')
    
    # saving param.
    parser.add_argument('--pt_save', type=int, default=20, help='model weight saving interval')
    parser.add_argument('--ckpt_save', type=int, default=10, help='checkpoint saving interval')
    parser.add_argument('--output', type=str, default='./output', help='training results saving path')
    parser.add_argument('--output_ckpt', type=str, default='./checkpoint', help='training results saving path')
    
    args = parser.parse_args()
    
    main(args)