# DCGAN
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ICLEVR_Dataset
from model.gan import Generator, Discriminator, weights_init

GPUS='0,1'
CUDA_VISIBLE_DEVICES=GPUS
ngpu=2

def main(args):
    # params.
    device = args.device
    
    # dataset
    dataset = ICLEVR_Dataset(args.tr_dir, args.tr_json, args.obj_json)
    dataloader = DataLoader(dataset, batch_size=args.b, num_workers=args.nw, shuffle=True)
    
    # model
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # checkpoints
    if args.use_ckpt:
        netG.load_state_dict(torch.load(args.ckpt_path))
        netD.load_state_dict(torch.load(args.ckpt_path))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)
    
    # loss
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # training loop
    loss = []
    for epoch in range(args.epochs):
        loss = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    
    # dataset
    parser.add_argument('--tr_dir', type=str, default='./dataset/iclevr', help='training data directory')
    parser.add_argument('--tr_json', type=str, default='./dataset/train.json', help='training data file')
    parser.add_argument('--obj_json', type=str, default='./dataset/objects.json', help='objects index')
    parser.add_argument('--eval_json', type=str, default='./dataset/test.json', help='validation data file')
    
    # training hyperparam.
    parser.add_argument('--device', type=str, default='cuda', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=12, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size of data')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='maximum epoch to train')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--use_ckpt', action='store_true', help='use checkpoint for training')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/epoch=20.ckpt')
    
    # saving param.
    parser.add_argument('--pt_save', type=int, default=10, help='model weight saving interval')
    parser.add_argument('--ckpt_save', type=int, default=5, help='checkpoint saving interval')
    parser.add_argument('--output_root', type=str, default='./output', help='training results saving path')
    
    args = parser.parse_args()
    
    main(args)