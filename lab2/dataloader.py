import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import  (Compose, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,
                                        ColorJitter, ToTensor, ConvertImageDtype, Normalize)
from logger import logger

def getData(root, mode):
    df = pd.read_csv(root + f'/{mode}.csv')
    path = df['filepaths'].tolist()
    label = df['label_id'].tolist()
    return path, label

class ButterflyMothDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.img_path, self.label = getData(root, mode)
        
        if mode == 'train':
            self.transform = Compose([
                RandomRotation(30),
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                ColorJitter(0.5, 0.5, 0.5, 0.25),
                ToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'valid' or mode == 'test':
            self.transform = Compose([
                ToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        logger.info("> Found %d images (%s)..." % (len(self.img_path), mode))  

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        # read image and label
        img = Image.open(self.root + '/' + self.img_path[index])
        label = self.label[index]
        
        # data augmentation
        img = self.transform(img)
        
        return img, label

class DataHandler:
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset
        
    def get_dataset(self, mode):
        if self.dataset == 'ButterflyMoth':
            return ButterflyMothDataset(self.root, mode)
    
    def get_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    dataHandler = DataHandler('./dataset', 'ButterflyMoth')
    train_dataset = dataHandler.get_dataset('valid')
    train_dataloader = dataHandler.get_dataloader(train_dataset, batch_size=16, num_workers=1)
    
    for data in train_dataloader:
        print(data)
        break
