import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import time
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from logger import logger
from VGG19 import vgg19
from ResNet50 import resnet50
from dataloader import DataHandler


def evaluate(model, cfg, valid_dataset, valid_dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in valid_dataloader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            total_loss += loss
            total_acc += (torch.max(y_pred, 1)[1] == y).sum().item()

    valid_loss = total_loss / len(valid_dataset)
    valid_acc = total_acc / len(valid_dataset)
    
    return valid_loss, valid_acc

def test(model, datahandler, cfg):
    # test set
    test_dataset = datahandler.get_dataset('test')
    test_dataloader = datahandler.get_dataloader(test_dataset, cfg.batch_size, cfg.num_workers)
    
    # model
    model.load_state_dict(torch.load(cfg.weight, map_location='cpu'))
    model.eval()
    
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # testing loop
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x, y = x.to(cfg.device), y.to(cfg.device)
            
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            total_loss += loss
            total_acc += (torch.max(y_pred, 1)[1] == y).sum().item()
        
    test_loss = total_loss / len(test_dataset)
    test_acc = total_acc / len(test_dataset)
    
    logger.info('> Testing Loss - {:4f}'.format(test_loss))
    logger.info('> Testing Accuracy - {}/{} = {:4f} = {}%'.format(int(total_acc), len(test_dataset), test_acc, test_acc*100))

        

def train(model, datahandler, cfg):
    # initialize
    epoch_init = 1
    best_val = 0.0
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []
    
    os.makedirs('./weight', exist_ok=True)
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    
    # train set
    train_dataset = datahandler.get_dataset('train')
    train_dataloader = datahandler.get_dataloader(train_dataset, cfg.batch_size, cfg.num_workers)
    
    # valid set
    valid_dataset = datahandler.get_dataset('valid')
    valid_dataloader = datahandler.get_dataloader(valid_dataset, cfg.batch_size, cfg.num_workers)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # use checkpoint
    if cfg.ckpt_use == True:
        try:
            ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if cfg.history == True:
                epoch_init = ckpt['epoch'] + 1
                train_epoch_loss = ckpt['train_loss']
                train_epoch_acc = ckpt['train_acc']
                val_epoch_loss = ckpt['valid_loss']
                val_epoch_acc = ckpt['valid_acc']
            logger.info(f'> Using {cfg.ckpt_path}')
            logger.info(f'> {"Keep" if cfg.history else "Clean"} history -> start training from {epoch_init} epoch')
        except Exception as e:
            logger.info(e)
    else:
        logger.info(f'> No using checkpoint')
    
    # training loop
    train_start = time.time()
    for epoch in range(epoch_init, cfg.epoch_nums + 1):   
        # train
        epoch_start = time.time()
        model.train()
        total_loss, total_acc = 0.0, 0.0   
        for x, y in tqdm(train_dataloader):
            x, y = x.to(cfg.device), y.to(cfg.device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # performance
            total_loss += loss
            total_acc += (torch.max(y_pred, 1)[1] == y).sum().item()
        
        training_loss = total_loss / len(train_dataset)
        training_acc = total_acc / len(train_dataset)
        
        train_epoch_loss.append(training_loss)
        train_epoch_acc.append(training_acc)

        # validate
        valid_loss, valid_acc = evaluate(model, cfg, valid_dataset, valid_dataloader, criterion)
        val_epoch_loss.append(valid_loss)
        val_epoch_acc.append(valid_acc)
        
        logger.info('Epoch {} > training loss: {:.4f}, training acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}, duration: {:.4f}'.format(epoch, training_loss, training_acc, valid_loss, valid_acc, time.time() - epoch_start))
    
        if epoch % cfg.save_weight == 0:
            weight_path = f'./weight/{cfg.model}_{epoch}.pt'
            torch.save(model.state_dict(), weight_path)
            logger.info(f'> Save the model weight at {weight_path}')
            
            if valid_acc > best_val:
                best_val = valid_acc
                torch.save(model.state_dict(), f'./weight/best_{cfg.model}.pt')
                logger.info('> Save the best model weight in epoch {} - val_acc: {:.4f}'.format(epoch, best_val))
            
            plot_path = './log' + f'/{cfg.model}_epoch{epoch}_acc.png'
            plt.title(f'{cfg.model} performance'), plt.ylabel('accuracy'), plt.xlabel('epoch')
            plt.plot(range(1, epoch + 1), train_epoch_acc, 'b', label='Training acc')
            plt.plot(range(1, epoch + 1), val_epoch_acc, 'r', label='Validation acc')
            plt.legend(loc='lower right')
            plt.savefig('./log' + f'/{cfg.model}_epoch{epoch}_acc.png')
            plt.clf()
            logger.info(f'> Save the plot of the accuracy in {plot_path}')
        
        if epoch % cfg.save_ckpt == 0:
            path = f'./checkpoint/{cfg.model}_epoch{epoch}.ckpt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'train_acc': train_epoch_acc,
                'valid_loss': val_epoch_loss,
                'valid_acc': val_epoch_acc,
            }, path)
            logger.info(f'> Save the checkpoint in {path}')
    
    logger.info('Total training time: {:.4f}'.format(time.time() - train_start))

def main(cfg):
    logger.info(f'> Mode - {cfg.mode}')
    if cfg.mode == 'train':
        logger.info(f'> Config: model-{cfg.model}, ckpt_use-{cfg.ckpt_use}, ckpt_path-{cfg.ckpt_path}, lr-{cfg.lr}, batch_size-{cfg.batch_size}, num_workers-{cfg.num_workers}')
    elif cfg.mode == 'test':
        logger.info(f'> Config: model-{cfg.model}, weight-{cfg.weight}')
    
    # datahandler
    datahandler = DataHandler(cfg.root, cfg.dataset)
    
    # model setting
    model = vgg19() if cfg.model == 'vgg19' else resnet50()
    model = model.to(device=cfg.device)
    
    if cfg.mode == 'train':
        train(model=model, datahandler=datahandler, cfg=cfg)
    elif cfg.mode == 'test':
        test(model=model, datahandler=datahandler, cfg=cfg)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='model mode: "train", "test"')
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg19', 'resnet50'], help='model type: "vgg19", "resnet50"')
    parser.add_argument('--weight', type=str, default='./weight/resnet50_500.pt', help='model weight path')
    
    # dataset
    parser.add_argument('--root', type=str, default='./dataset', help='the root of dataset')
    parser.add_argument('--dataset', type=str, default='ButterflyMoth', help='dataset name')
    
    # training
    parser.add_argument('--ckpt_use', action='store_true', help='use checkpoint for training or not')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/resnet50_epoch100.ckpt', help='checkpoint path')
    parser.add_argument('--history', action='store_true', help='keep loss history')
    
    parser.add_argument('--device', type=str, default='cuda:0', help='training device')
    parser.add_argument('--epoch_nums', type=int, default=500, help='max number of epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='numbers of workers')
    
    parser.add_argument('--save_weight', type=int, default=50, help='epoch interval of saving model weight')
    parser.add_argument('--save_ckpt', type=int, default=20, help='epoch interval of saving checkpoint')
    parser.add_argument('--val_interval', type=int, default=5, help='epoch interval of running validation set')
    
    args = parser.parse_args()

    main(args)
