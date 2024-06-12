# inference diffusion model
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from data import ICLEVR_Dataset
from model.unet import Unet
from model.ddpm import DDPM
from evaluator import evaluation_model


def main(args):
    # params.
    device = args.device
    os.makedirs(args.output_path, exist_ok=True)
    
    # dataset
    test_dataset = ICLEVR_Dataset(args.root, test_file=args.test_file, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    new_test_dataset = ICLEVR_Dataset(args.root, test_file=args.new_test_file, mode='test')
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    # model
    evaluator = evaluation_model(device)
    unet_model = Unet().to(args.device)
    ddpm = DDPM(unet_model=unet_model, betas=(args.beta_start, args.beta_end), noise_steps=args.step, device=args.device).to(device)

    # test set
    acc = 0.0
    ddpm.load_state_dict(torch.load(args.weight_t))
    
    # The sampling process is stochastic, with an accuracy range of 0.65 to 0.82.
    # It can't guarantee the accuracy always larger than 0.8
    acc, grid = test(ddpm, evaluator, test_dataloader, device)
    path = os.path.join(args.output_path, 'test.png')
    save_image(grid, path)
    print(f'Test set accuracy={acc:.4f}')
    
    # new test set
    acc = 0.0
    ddpm.load_state_dict(torch.load(args.weight_nt))
    
    # The sampling process is stochastic, with an accuracy range of 0.65 to 0.82.
    # It can't guarantee the accuracy always larger than 0.8
    acc, grid = test(ddpm, evaluator, new_test_dataloader, device)
    path = os.path.join(args.output_path, 'new_test.png')
    save_image(grid, path)
    print(f'New test set accuracy={acc:.4f}')
    

def test(ddpm, evaluator, test_dataloader, device):
    ddpm.eval()
    x_gen, label = [], []
        
    with torch.no_grad():
        for cond in test_dataloader:
            cond = cond.to(device)
            x_i = ddpm.sample(cond, (3, 64, 64), device)
            x_gen.append(x_i)
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
    parser.add_argument('--step', type=int, default=300, help='timestep to process')
    parser.add_argument('--beta_start', type=int, default=1e-4, help='start beta value')
    parser.add_argument('--beta_end', type=int, default=0.02, help='end beta value')
    parser.add_argument('--device', type=str, default='cuda:0', help='device you choose to use')
    parser.add_argument('--num_workers', '-nw',type=int, default=8, help='numbers of workers')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size of data')
    
    # inference 
    parser.add_argument('--weight_t', type=str, default='./output2/dm/ddpm_best.pth', help='test set model weight')
    parser.add_argument('--weight_nt', type=str, default='./output/dm/ddpm_new_best.pth', help='new test model weight')
    parser.add_argument('--output_path', type=str, default='./results/dm', help='infernece results saving path')
    
    args = parser.parse_args()
    
    main(args)