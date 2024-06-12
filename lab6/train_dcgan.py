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
    ckpt_save_path = args.output_ckpt +'/' + 'dcgan'
    output_path = args.output + '/' + 'dcgan'
    os.makedirs(ckpt_save_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # dataset
    train_dataloader = DataLoader(
        ICLEVR_Dataset(args.root), 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        ICLEVR_Dataset(args.root, test_file=args.test_file, mode='test'), 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False
    )
    
    new_test_dataloader = DataLoader(
        ICLEVR_Dataset(args.root, test_file=args.new_test_file, mode='test'), 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # model
    netG = Generator(nz=args.Z_dims, ngf=args.G_dims, nc=args.c_dims, n_classes=args.num_class).to(device)
    netD = Discriminator(n_classes=args.num_class, ndf=args.D_dims, img_size=64).to(device)
    
    # loss
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', factor=0.95, patience=5, min_lr=0)
    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='max', factor=0.95, patience=5, min_lr=0)
    evaluator = evaluation_model(device)
    
    # checkpoints
    if args.use_ckpt:
        print(f'Using checkpoint: {args.ckpt_G_path}, {args.ckpt_D_path}')
        ckpt_G = torch.load(args.ckpt_G_path)
        ckpt_D = torch.load(args.ckpt_D_path)
        netG.load_state_dict(ckpt_G['netG'])
        netD.load_state_dict(ckpt_D['netD'])
        optimizerG.load_state_dict(ckpt_G['optimG'])
        optimizerD.load_state_dict(ckpt_D['optimD'])
    else:
        print('>>> Train from scratch <<<')
        netG.apply(weights_init)
        netD.apply(weights_init)
    
    
    
    # training loop
    train(args, netG, netD, optimizerG, optimizerD, schedulerD, schedulerG, criterion, evaluator, train_dataloader, test_dataloader, new_test_dataloader, device, ckpt_save_path, output_path)
    

def train(args, netG, netD, optimG, optimD, schedulerD, schedulerG, criterion, evaluator, train_dataloader, test_dataloader, new_test_dataloader, device, ckpt_save_path, output_path):
    best_acc, new_best_acc = 0.0, 0.0
    acc_list, new_acc_list = [], []
    loss_d_total, loss_g_total = [], []
    
    for epoch in range(1, args.epochs+1):
        netD.train()
        netG.train()
        loss_d_epoch, loss_g_epoch = 0.0, 0.0
        
        for image, cond in (pbar := tqdm(train_dataloader)):
            optimD.zero_grad()
            optimG.zero_grad()
            batch_size = image.size(0)
            cond = cond.to(device)
            
            # real image & label
            real_image = image.to(device)
            real_y = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)
            
            # fake image & label
            noise = torch.randn(batch_size, args.Z_dims, 1, 1, device=device)
            fake_image = netG(noise, cond)
            fake_y = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)
            
            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            step, real, fake = 0.0, 0.0, 0.0
            for _ in range(args.ratio):
                optimD, loss_D, loss_D_real, loss_D_fake, D_x, D_G_z1 = train_discriminator_one_step(netD, optimD, cond, real_image, real_y, fake_image.detach(), fake_y, criterion)
                step += loss_D.item()
                real += loss_D_real.item()
                fake += loss_D_fake.item()
            step /= args.ratio
            real /= args.ratio
            fake /= args.ratio
            loss_d_epoch += step
            
            # Update Generator: maximize log(D(G(z)))
            optimG, loss_G, D_G_z2 = train_generator_one_step(netD, optimG, cond, fake_image, batch_size, criterion, device)
            loss_g_epoch += loss_G.item()
            desp = f'Train - Epoch {epoch}/{args.epochs} - lr: {schedulerG.get_last_lr()[0]}, Loss_G: {loss_G.item():.4f}, Loss_D: {step:.4f}, real: {real:.4f}, fake: {fake:.4f}, D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}'
            tqdm_bar(pbar, desp)
            
        
        loss_d_epoch /= len(train_dataloader)
        loss_d_total.append(loss_d_epoch)
        avg_loss_D = sum(loss_d_total)/len(loss_d_total)
        schedulerD.step(best_acc + new_best_acc)

        loss_g_epoch /= len(train_dataloader)
        loss_g_total.append(loss_g_epoch)
        avg_loss_G = sum(loss_g_total)/len(loss_g_total)
        schedulerG.step(best_acc + new_best_acc)

        if epoch % args.ckpt_save == 0:
            path = os.path.join(output_path, f'Epoch_{epoch}')
            os.makedirs(path, exist_ok=True)

            # save discriminator loss curve
            title, curve1, curve2 = f'Training_Epoch_{epoch}_BCELoss', f'Discriminator - avg. loss={avg_loss_D:.3f}', f'Generator - avg. loss={avg_loss_G:.3f}'
            plot_curve(title, len(loss_d_total), loss_d_total, curve1, loss_g_total, curve2, path=os.path.join(path, 'Loss.png'))
            
            # validation - test file
            acc, grid = valid(args, netG, netD, evaluator, test_dataloader, device)
            acc_list.append(acc)
            save_image(grid, os.path.join(path, 'test.png'))
            if acc > best_acc:
                best_acc = acc
                torch.save(netG.state_dict(), os.path.join(output_path, 'netG_best.pth'))
                torch.save(netD.state_dict(), os.path.join(output_path, 'netD_best.pth'))
                print(f'>>> Best Acc: {best_acc}')
                print(f'> Save weight at', os.path.join(output_path, 'netG_best.pth'))
                print(f'> Save weight at', os.path.join(output_path, 'netD_best.pth'))
            
            # validation - new test file
            acc, grid = valid(args, netG, netD, evaluator, new_test_dataloader, device)
            new_acc_list.append(acc)
            save_image(grid, os.path.join(path, 'new_test.png'))
            if acc > new_best_acc:
                new_best_acc = acc
                torch.save(netG.state_dict(), os.path.join(output_path, 'netG_new_best.pth'))
                torch.save(netD.state_dict(), os.path.join(output_path, 'netD_new_best.pth'))
                print(f'>>> New Best Acc: {new_best_acc}')
                print(f'> Save weight at', os.path.join(output_path, 'netG_new_best.pth'))
                print(f'> Save weight at', os.path.join(output_path, 'netD_new_best.pth'))
            
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
    

def train_discriminator_one_step(netD, optimD, cond, real_image, real_y, fake_image, fake_y, criterion):
    optimD.zero_grad()
    
    # occasionally flip the labels when training the discriminator
    if random.random() < 0.1:
        real_y, fake_y = fake_y, real_y
    
    # train with real
    output = netD(real_image, cond)
    D_x = output.mean().item()
    loss_D_real = criterion(output, real_y)
    
    # train with fake
    output = netD(fake_image, cond)
    D_G_z1 = output.mean().item()
    loss_D_fake = criterion(output, fake_y)
    
    # total loss
    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optimD.step()
    
    return optimD, loss_D, loss_D_real, loss_D_fake, D_x, D_G_z1


def train_generator_one_step(netD, optimG, cond, fake_image, batch_size, criterion, device):
    optimG.zero_grad()
    
    generator_label = torch.ones(batch_size).to(device)  # fake labels are real for generator cost
    
    output = netD(fake_image, cond)
    D_G_z2 = output.mean().item()
    loss_G = criterion(output, generator_label)
    
    loss_G.backward()
    optimG.step()
    
    return optimG, loss_G, D_G_z2


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
    parser.add_argument('--num_class', type=int, default=24, help='numbers of class')
    parser.add_argument('--c_dims', type=int, default=100, help='condition dimension')
    parser.add_argument('--Z_dims', type=int, default=100, help='latent dimension')
    parser.add_argument('--G_dims', type=int, default=300, help='generator feature dimension')
    parser.add_argument('--D_dims', type=int, default=100, help='discriminator feature dimension')
    parser.add_argument('--ratio', type=int, default=1, help='update ratio, netD : netG')
    
    # training hyperparam.
    parser.add_argument('--device', type=str, default='cuda:0', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=8, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size of data')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='maximum epoch to train')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--use_ckpt', action='store_true', help='use checkpoint for training')
    parser.add_argument('--ckpt_G_path', type=str, default='./checkpoint/dcgan/netG_Epoch_200.ckpt')
    parser.add_argument('--ckpt_D_path', type=str, default='./checkpoint/dcgan/netD_Epoch_200.ckpt')
    
    # saving param.
    parser.add_argument('--pt_save', type=int, default=50, help='model weight saving interval')
    parser.add_argument('--ckpt_save', type=int, default=25, help='checkpoint saving interval')
    parser.add_argument('--output', type=str, default='./output', help='training results saving path')
    parser.add_argument('--output_ckpt', type=str, default='./checkpoint', help='training results saving path')
    
    args = parser.parse_args()
    
    main(args)