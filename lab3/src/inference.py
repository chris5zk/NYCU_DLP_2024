import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import DICEloss
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import resnet34_unet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    
    parser.add_argument('--model', default='./saved_models/weight/best_unet.pth', help='path to the stored model weight')
    
    parser.add_argument('--data_path', default='./dataset/oxford-iiit-pet', type=str, help='path to the input data')
    parser.add_argument('--data_augmentation', '-da', action='store_true', help='use data augmentation or not')
    
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='device for testing')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # dataset
    test_dataset = load_dataset(args.data_path, 'test', data_augentation=args.data_augmentation)
    test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=0, shuffle=False)

    # read model
    model = UNet()
    model.load_state_dict(torch.load(args.model))
    
    model = model.to(args.device)
    model.eval()
    
    # criterion
    criterion = DICEloss()
    
    # testing loop
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for sample in tqdm(test_dataloader):
            x, y = sample['image'].to(args.device), sample['mask'].to(args.device)

            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            score = 1 - loss
            
            total_loss += loss
            total_acc += score
            
            img = (y_pred.cpu().detach().detach().numpy() > 0.5) * 1
            image = np.expand_dims(img.squeeze(), axis=-1)
            print(f'loss: {loss}, acc: {score}')
            plt.imshow(image)
            plt.show()

    test_loss = total_loss / math.ceil(len(test_dataset) / args.batch_size)
    test_acc = total_acc / math.ceil(len(test_dataset) / args.batch_size)      
    
    print('> Testing Loss - {:.4f}'.format(test_loss))
    print('> Testing Accuracy - {:.4f} = {:.2f}%'.format(test_acc, test_acc*100))  