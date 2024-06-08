import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  (Compose, ToTensor, Normalize, Resize)


class ICLEVR_Dataset(Dataset):
    def __init__(self, root, test_file=None, mode='train'):
        self.mode = mode
        self.root = root
        self.test_file = test_file

        # get image path and label
        self.img_list, self.label_list = self.get_data()
        self.transform = self.mode_transform()
    
    def __len__(self):
        return len(self.label_list)
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            # read img from list
            img = Image.open(os.path.join(self.root, 'iclevr', self.img_list[idx])).convert('RGB')
            img = self.transform(img)
            label = torch.Tensor(self.label_list[idx])
            return img, label
        else:
            label = torch.Tensor(self.label_list[idx])
            return label
    
    def get_data(self):
        # init setting
        img_list = None
        object_class_id = json.load(open(os.path.join(self.root, 'objects.json')))
    
        if self.mode == 'train':
            # read data
            train_data_info = json.load(open(os.path.join(self.root, 'train.json')))
            img_list        = list(train_data_info.keys())
            label_list_org  = list(train_data_info.values())
            
            # change label to one-hot vector
            label_list = []
            for objects in label_list_org:
                one_hot_vector = np.zeros(len(object_class_id), dtype=np.int32)
                for obj in objects:
                    one_hot_vector[object_class_id[obj]] = 1
                label_list.append(one_hot_vector)
            label_list = np.array(label_list)
        else:
            # read data
            test_data = json.load(open(os.path.join(self.root, self.test_file)))
            
            # change label to one-hot vector
            label_list = []
            for objects in test_data:
                one_hot_vector = np.zeros(len(object_class_id), dtype=np.int32)
                for obj in objects:
                    one_hot_vector[object_class_id[obj]] = 1
                label_list.append(one_hot_vector)
            label_list = np.array(label_list)            
        
        return img_list, label_list
        
    def mode_transform(self):
        if self.mode == 'train':
            transform = Compose([
                Resize((64, 64)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = Compose([
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
        return transform

        
if __name__ == '__main__':

    root = './dataset'
    dataset = ICLEVR_Dataset(root)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=8)
    
    print(len(dataset))
    print(len(dataloader))
    for data in tqdm(dataloader):
        print(data[0].shape)
        print(data[1].shape)
        break