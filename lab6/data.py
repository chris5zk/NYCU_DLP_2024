import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  (Compose, ToTensor, Normalize, Resize)


class ICLEVR_Dataset(Dataset):
    def __init__(self, train_dir, tr_info_file, obj_json, test_json, mode='train'):
        self.mode = mode
        self.train_dir = train_dir
        self.tr_info_file = tr_info_file
        
        with open(obj_json, 'r') as f:
            self.obj_dict = json.load(f)
            
        with open(test_json, 'r') as f:
            self.test_list = json.load(f)
        
        self.img_list, self.label_list = self.get_data(mode)
        self.transform = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list)
        else:
            return len(self.test_list)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            # read img from list
            path = os.path.join(self.train_dir, self.img_list[index])
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            
            # read label from list
            label = self.label_list[index]
        
            return img, label
        
        else:
            return self.label_list[index]
    
    def get_data(self, mode):
        
        if mode == 'train':
            # read the images and labels
            with open(self.tr_info_file, 'r') as f:
                data_info = json.load(f)
                img_list = list(data_info.keys())
                org_label_list = list(data_info.values())
            
            # change label to one-hot vector
            label_list = []
            for idx, objects in enumerate(org_label_list):
                label_list.append(torch.zeros(len(self.obj_dict)))
                for object in objects:
                    label_list[idx][self.obj_dict[object]] = 1
                    
            return img_list, label_list
        
        elif mode == 'val' or 'test':
            label_list = []
            for idx, objects in enumerate(self.test_list):
                label_list.append(torch.zeros(len(self.obj_dict)))
                for object in objects:
                    label_list[idx][self.obj_dict[object]] = 1
                    
            return [], label_list
        
        

        
if __name__ == '__main__':
    train_data = './dataset/iclevr'
    train_json = './dataset/train.json'
    object_idx = './dataset/objects.json'
    
    # dataset = ICLEVR_Dataset(train_data, train_json, object_idx)
    
    # dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
    
    # print(len(dataloader))
    # for data in tqdm(dataloader):
    #     print(data)
    #     break
    
    with open('./dataset/test.json', 'r') as f:
        test_json = json.load(f)
    with open('./dataset/new_test.json', 'r') as f:
        new_test_json = json.load(f)
        
    print(len(test_json))
    print(len(new_test_json))