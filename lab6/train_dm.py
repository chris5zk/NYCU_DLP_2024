import torch
import argparse

from torch.utils.data import DataLoader

from data import ICLEVR_Dataset



def main(args):
    # dataset
    dataset = ICLEVR_Dataset(args.tr_dir, args.tr_json, args.obj_json)
    dataloader = DataLoader(dataset, batch_size=args.b, num_workers=args.nw, shuffle=True)
    
    # model
    
    # loss
    
    # training loop
    
    
    # 
    
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    
    # dataset
    parser.add_argument('--tr_dir', type=str, default='./dataset/iclevr', help='training data directory')
    parser.add_argument('--tr_json', type=str, default='./dataset/train.json', help='training data file')
    parser.add_argument('--obj_json', type=str, default='./dataset/objects.json', help='objects index')
    parser.add_argument('--eval_json', type=str, default='./dataset/test.json', help='validation data file')
    
    # training hyperparam.
    parser.add_argument('--device', type=str, default='cuda:0', help='device you choose to use')
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