import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from data import ICLEVR_Dataset
from model.dcgan import Generator, Discriminator
from evaluator import evaluation_model


def main(args):
    # params.
    device = args.device
    os.makedirs(args.output_path, exist_ok=True)
    root = args.weight_path
    
    # dataset
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
    evaluator = evaluation_model(device)
    netG = Generator(nz=args.Z_dims, ngf=args.G_dims, nc=args.c_dims, n_classes=args.num_class).to(device)
    netD = Discriminator(n_classes=args.num_class, ndf=args.D_dims, img_size=64).to(device)
    
    # test set
    acc = 0.0
    netG.load_state_dict(torch.load(os.path.join(root, 'netG_best.pth')))
    netD.load_state_dict(torch.load(os.path.join(root, 'netD_best.pth')))
    
    acc, grid = test(args, netG, netD, evaluator, test_dataloader, device)
    path = os.path.join(args.output_path, 'test.png')
    save_image(grid, path)
    
    print(f'Test - Accuracy: {acc:.4f}')
    
    
    # new test set
    acc = 0.0
    netG.load_state_dict(torch.load(os.path.join(root, 'netG_new_best.pth')))
    netD.load_state_dict(torch.load(os.path.join(root, 'netD_new_best.pth')))
    
    acc, grid = test(args, netG, netD, evaluator, new_test_dataloader, device)
    path = os.path.join(args.output_path, 'new_test.png')
    save_image(grid, path)
    
    print(f'New Test - Accuracy: {acc:.4f}')


def test(args, netG, netD, evaluator, test_dataloader, device):
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
        
    return acc, grid


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train")
    
    # dataset
    parser.add_argument('--root', type=str, default='./dataset', help='training data directory')
    parser.add_argument('--test_file', type=str, default='test.json', help='testing data file')
    parser.add_argument('--new_test_file', type=str, default='new_test.json', help='new testing data file')
    
    # model
    parser.add_argument('--model', type=str, default='dcgan', help='model type')
    parser.add_argument('--num_class', type=int, default=24, help='numbers of class')
    parser.add_argument('--c_dims', type=int, default=100, help='condition dimension')
    parser.add_argument('--Z_dims', type=int, default=100, help='latent dimension')
    parser.add_argument('--G_dims', type=int, default=300, help='generator feature dimension')
    parser.add_argument('--D_dims', type=int, default=100, help='discriminator feature dimension')

    # inference 
    parser.add_argument('--device', type=str, default='cuda:1', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=8, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size of data')
    
    # path
    parser.add_argument('--weight_path', type=str, default='./output/dcgan', help='GAN weight root path')
    parser.add_argument('--output_path', type=str, default='./results/dcgan', help='infernece results saving path')
    
    args = parser.parse_args()
    
    main(args)