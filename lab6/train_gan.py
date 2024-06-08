# DCGAN
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
from model.dcgan import Generator, Discriminator


CUDA_VISIBLE_DEVICES='0'

# custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def main(args):
    # params.
    device = args.device
    ckpt_save_path = args.output_ckpt +'/' + 'gan'
    output_path = args.output + '/' + 'gan'
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
        print(f'Using checkpoint: {args.ckpt_path}')
        netG.load_state_dict(torch.load(args.ckpt_path))
        netD.load_state_dict(torch.load(args.ckpt_path))
    else:
        print('Intialize the model weight.')
        netG.apply(weights_init)
        netD.apply(weights_init)
    
    # loss
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    evaluator = evaluation_model(device)
    
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
        
        for image, cond in (pbar := tqdm(train_dataloader)):
            batch_size = image.size(0)
            cond = cond.to(device)
            
            # real image & label
            real_image = image.to(device)
            real_y = torch.ones(batch_size, dtype=torch.float32).to(device)
            
            # fake image & label
            noise = torch.randn(batch_size, args.Z_dims, 1, 1, device=device)
            fake_image = netG(noise, cond).detach()
            fake_y = torch.zeros(batch_size, dtype=torch.float32).to(device)
            
            if epoch % args.ratio == 0:
                # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                optimD, loss_D = train_discriminator_one_epoch(netD, optimD, cond, real_image, real_y, fake_image, fake_y, criterion, device)
                loss_d_epoch += loss_D.item()
                desp = f'Train Discriminator - Epoch {epoch}/{args.epochs}, Loss_D: {loss_D.item():.4f}'
                tqdm_bar(pbar, desp)
            else:
                # Update Generator: maximize log(D(G(z)))
                optimG, loss_G = train_generator_one_epoch(netD, optimG, cond, fake_image, batch_size, criterion, device)
                loss_g_epoch += loss_G.item()
                desp = f'Train Generator - Epoch {epoch}/{args.epochs}, Loss_G: {loss_G.item():.4f}'
                tqdm_bar(pbar, desp)
            
        if epoch % args.ratio == 0:
            loss_d_epoch /= len(train_dataloader)
            loss_d_total.append(loss_d_epoch)
            avg_loss_D = sum(loss_d_total)/len(loss_d_total)
        else:
            loss_g_epoch /= len(train_dataloader)
            loss_g_total.append(loss_g_epoch)
            avg_loss_G = sum(loss_g_total)/len(loss_g_total)

        if epoch % args.ckpt_save == 0:
            path = os.path.join(output_path, f'Epoch_{epoch}')
            os.makedirs(path, exist_ok=True)

            # save discriminator loss curve
            title, curve = f'Discriminator_Epoch_{epoch}_BCELoss', f'Discriminator - avg. loss={avg_loss_D:.3f}'
            plot_curve(title, len(loss_d_total), loss_d_total, curve, path=os.path.join(path, 'D_Loss.png'))
            
            # save generator loss curve
            title, curve = f'Generator_Epoch_{epoch}_BCELoss', f'Generator - avg. loss={avg_loss_G:.3f}'
            plot_curve(title, len(loss_g_total), loss_g_total, curve, path=os.path.join(path, 'G_Loss.png'))
            
            # validation - test file
            acc, grid = valid(args, netG, netD, evaluator, test_dataloader, device)
            acc_list.append(acc)
            save_image(grid, os.path.join(path, 'test.png'))
            if acc > best_acc:
                best_acc = acc
                torch.save(netG.state_dict(), os.path.join(path, 'netG_best.pth'))
                torch.save(netD.state_dict(), os.path.join(path, 'netD_best.pth'))
                print(f'>>> Best Acc: {best_acc}')
                print(f'> Save weight at', os.path.join(path, 'netG_best.pth'))
                print(f'> Save weight at', os.path.join(path, 'netD_best.pth'))
            
            # validation - new test file
            acc, grid = valid(args, netG, netD, evaluator, new_test_dataloader, device)
            new_acc_list.append(acc)
            save_image(grid, os.path.join(path, 'new_test.png'))
            if acc > new_best_acc:
                new_best_acc = acc
                torch.save(netG.state_dict(), os.path.join(path, 'netG_new_best.pth'))
                torch.save(netD.state_dict(), os.path.join(path, 'netD_new_best.pth'))
                print(f'>>> New Best Acc: {new_best_acc}')
                print(f'> Save weight at', os.path.join(path, 'netG_new_best.pth'))
                print(f'> Save weight at', os.path.join(path, 'netD_new_best.pth'))
            
            # plot accuracy curve
            title, curve1, curve2 = f'Train_Epoch_{epoch}_Acc', f'test set(Best={best_acc:.3f})', f'new test set(Best={new_best_acc:.3f})'
            plot_curve(title, len(acc_list), acc_list, curve1, new_acc_list, curve2, os.path.join(path, 'Accuracy.png'), loc='lower right')
            
            # save checkpoint
            path = os.path.join(ckpt_save_path, f'netG_Epoch_{epoch}.ckpt')
            torch.save({
                "netG":     netG.state_dict(),
                "optimG" :  optimG.state_dict(),
            }, path)
            print(f'> Save checkpoint at {path}')
            
            path = os.path.join(ckpt_save_path, f'netD_Epoch_{epoch}.ckpt')
            torch.save({
                "netD":     netD.state_dict(),
                "optimD" :  optimD.state_dict(),
            }, path)
            print(f'> Save checkpoint at {path}')
            
        if epoch % args.pt_save == 0:
            # save weights
            path = os.path.join(output_path, f'Epoch_{epoch}')
            torch.save(netG.state_dict(), os.path.join(path, f'netG_Epoch_{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(path, f'netD_Epoch_{epoch}.pth'))
            print(f'> Save weight at', os.path.join(path, f'netG_Epoch_{epoch}.pth'))
            print(f'> Save weight at', os.path.join(path, f'netD_Epoch_{epoch}.pth'))
    

def train_discriminator_one_epoch(netD, optimD, cond, real_image, real_y, fake_image, fake_y, criterion, device):
    optimD.zero_grad()
    
    # train with real
    output = netD(real_image, cond)
    loss_D_real = criterion(output, real_y)
    
    # train with fake
    output = netD(fake_image, cond)
    loss_D_fake = criterion(output, fake_y)
    
    # total loss
    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optimD.step()
    
    return optimD, loss_D


def train_generator_one_epoch(netD, optimG, cond, fake_image, batch_size, criterion, device):
    optimG.zero_grad()
    
    generator_label = torch.ones(batch_size).to(device)  # fake labels are real for generator cost
    output = netD(fake_image, cond)
    
    loss_G = criterion(output, generator_label)
    loss_G.backward()
    optimG.step()
    
    return optimG, loss_G


def valid(args, netG, netD, evaluator, test_dataloader, device):
    netG.eval()
    netD.eval()
    x_gen, label = [], []
    
    with torch.no_grad():
        for cond in test_dataloader:
            cond = cond.to(device)
            noise = torch.randn(cond.shape[0], args.Z_dims, 1, 1, device=device)
            gen_image = netG(noise, cond)
            x_gen.append(gen_image)
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
    parser.add_argument('--Z_dims', type=int, default=100, help='latent dimension')
    parser.add_argument('--G_dims', type=int, default=256, help='generator feature dimension')
    parser.add_argument('--D_dims', type=int, default=256, help='discriminator feature dimension')
    parser.add_argument('--ratio', type=int, default=3, help='update ratio, netG : netD')
    
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