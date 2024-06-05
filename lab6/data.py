import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  (Compose, ToTensor, Normalize, Resize)


class ICLEVR_Dataset(Dataset):
    def __init__(self, train_dir, tr_info_file, obj_json):
        self.train_dir = train_dir
        self.tr_info_file = tr_info_file
        
        with open(obj_json, 'r') as f:
            self.obj_dict = json.load(f)
            
        self.transform = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, index):
        # read img from list
        path = os.path.join(self.train_dir, self.img_list[index])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        # read label from list
        label = self.labels[index]
        
        return img, label
    
    def get_train_data(self):
        # read the images and labels
        with open(self.tr_info_file, 'r') as f:
            data_info = json.load(f)
            self.img_list = list(data_info.keys())
            label_list = list(data_info.values())
            
        # change label to one-hot vector
        self.labels = []
        for idx, objects in enumerate(label_list):
            self.labels.append(torch.zeros(len(self.obj_dict)))
            for object in objects:
                self.labels[idx][self.obj_dict[object]] = 1
                
        
if __name__ == '__main__':
    train_data = './dataset/iclevr'
    train_json = './dataset/train.json'
    object_idx = './dataset/objects.json'
    
    dataset = ICLEVR_Dataset(train_data, train_json, object_idx)
    dataset.get_train_data()
    
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
    
    for data in tqdm(dataloader):
        print(data)
        break