import dataset
import utils
from utils import EarlyStopping, LRScheduler
import os
import pandas as pd
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time

parser = argparse.ArgumentParser(description='PET lymphoma classification')

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output folder')

#MODEL PARAMS
parser.add_argument('--normalize', action='store_true', default=False, help='normalize images')
parser.add_argument('--checkpoint', default='', type=str, help='model checkpoint if any')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')

#OPTIMIZATION PARAMS
parser.add_argument('--optimizer', default='sgd', type=str, help='The optimizer to use (default: sgd)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--lr_anneal', type=int, default=15, help='period for lr annealing (default: 15). Only works for SGD')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

#TRAINING PARAMS
parser.add_argument('--seed_index', default=0, type=int, metavar='INT', choices=list(range(0,20)),help='which seed split index')   
parser.add_argument('--run', default=1, type=int, metavar='INT', help='repetition run with same settings')   
parser.add_argument('--batch_size', type=int, default=50, help='how many images to sample per slide (default: 50)')
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
parser.add_argument('--augm', default=0, type=int, choices=[0,1,2,3,12,4,5,14,34,45], help='augmentation procedure 0=none,1=flip,2=rot,3=flip LR, 12=flip+rot, 4=scale, 5=noise, 14=flip+scale, 34=flipLR+scale, 45=scale+noise (default: 0)')
parser.add_argument('--balance', action='store_true', default=False, help='balance dataset')
parser.add_argument('--lr_scheduler', action='store_true',default=False, help='decrease LR on platau')
parser.add_argument('--early_stopping', action='store_true',default=False, help='use early stopping')

def main():
    ### Get user input
    global args
    args = parser.parse_args()
    print(args)
    best_auc = 0.

    ### Output directory and files
    if not os.path.isdir(args.output):
        try:
            os.mkdir(args.output)
        except OSError:
            print ('Creation of the output directory "{}" failed.'.format(args.output))
        else:
            print ('Successfully created the output directory "{}".'.format(args.output))
    
    ### Get model
    model = utils.get_model()
    if args.checkpoint:
        ch = torch.load(args.checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if args.resume:
        ch = torch.load( os.path.join(args.output,'checkpoint_seed'+str(args.seed_index)+'_run'+str(args.run)+'.pth') )
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    ### Set optimizer
    optimizer = utils.create_optimizer(model, args.optimizer, args.lr, args.momentum, args.wd)
    if args.resume and 'optimizer' in ch:
        optimizer.load_state_dict(ch['optimizer'])
        print('Loaded optimizer state')
    cudnn.benchmark = True
    
    ### Augmentations
    flipHorVer = dataset.RandomFlip()
    flipLR     = dataset.RandomFlipLeftRight()
    rot90      = dataset.RandomRot90()
    scale      = dataset.RandomScale()
    noise      = dataset.RandomNoise()
    if args.augm==0:
        transform = None
    elif args.augm==1:
        transform = transforms.Compose([flipHorVer])
    elif args.augm==2:
        transform = transforms.Compose([rot90])
    elif args.augm==3:
        transform = transforms.Compose([flipLR])
    elif args.augm==12:
        transform = transforms.Compose([flipHorVer,rot90])
    elif args.augm==4:
        transform = transforms.Compose([scale])
    elif args.augm==5:
        transform = transforms.Compose([noise])
    elif args.augm==14:
        transform = transforms.Compose([flip,scale])
    elif args.augm==34:
        transform = transforms.Compose([flipLR,scale])
    elif args.augm==45:
        transform = transforms.Compose([scale,noise])
    
    ### Set datasets
    train_dset,trainval_dset,val_dset,_,balance_weight_neg_pos = dataset.get_datasets_singleview(transform,args.normalize,args.balance,args.seed_index)
    print('Datasets train:{}, val:{}'.format(len(train_dset.df),len(val_dset.df))) 
    
    ### Set loss criterion
    if args.balance:
        w = torch.Tensor(balance_weight_neg_pos)
        print('Balance loss with weights:',balance_weight_neg_pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
    
    ### Early stopping
    if args.lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
        if args.resume and 'lr_scheduler' in ch:
            lr_scheduler.lr_scheduler.load_state_dict(ch['lr_scheduler'])
            print('Loaded lr_scheduler state')
    if args.early_stopping:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        if args.resume and 'early_stopping' in ch:
            early_stopping.best_loss = ch['early_stopping']['best_loss']
            early_stopping.counter = ch['early_stopping']['counter']
            early_stopping.min_delta = ch['early_stopping']['min_delta']
            early_stopping.patience = ch['early_stopping']['patience']
            early_stopping.early_stop = ch['early_stopping']['early_stop']
            print('Loaded early_stopping state')
        
    ### Set loaders
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    trainval_loader = torch.utils.data.DataLoader(trainval_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    ### Set output files
    convergence_name = 'convergence_seed'+str(args.seed_index)+'_run'+str(args.run)+'.csv'
    if not args.resume:
        fconv = open(os.path.join(args.output,convergence_name), 'w')
        fconv.write('epoch,split,metric,value\n')
        fconv.close()
    
    ### Main training loop
    if args.resume:
        epochs = range(ch['epoch']+1,args.nepochs+1)
    else:
        epochs = range(args.nepochs+1)
    
    for epoch in epochs:
        if args.optimizer == 'sgd':
            utils.adjust_learning_rate(optimizer, epoch, args.lr_anneal, args.lr)
        
        ### Training logic
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion, optimizer)
        else:
            loss = np.nan
        ### Printing stats
        fconv = open(os.path.join(args.output,convergence_name), 'a')
        fconv.write('{},train,loss,{}\n'.format(epoch, loss))
        fconv.close()
        
        ### Validation logic
        # Evaluate on train data
        train_probs = test(epoch, trainval_loader, model)
        train_auc, train_ber, train_fpr, train_fnr = train_dset.errors(train_probs)
        # Evaluate on validation set
        val_probs = test(epoch, val_loader, model)
        val_auc, val_ber, val_fpr, val_fnr = val_dset.errors(val_probs)
        
        print('Epoch: [{}/{}]\tLoss: {:.6f}\tAUC: {:.4f}\t{:.4f}'.format(epoch, args.nepochs, loss, train_auc, val_auc))
        
        fconv = open(os.path.join(args.output,convergence_name), 'a')
        fconv.write('{},train,auc,{}\n'.format(epoch, train_auc))
        fconv.write('{},train,ber,{}\n'.format(epoch, train_ber))
        fconv.write('{},train,fpr,{}\n'.format(epoch, train_fpr))
        fconv.write('{},train,fnr,{}\n'.format(epoch, train_fnr))
        fconv.write('{},validation,auc,{}\n'.format(epoch, val_auc))
        fconv.write('{},validation,ber,{}\n'.format(epoch, val_ber))
        fconv.write('{},validation,fpr,{}\n'.format(epoch, val_fpr))
        fconv.write('{},validation,fnr,{}\n'.format(epoch, val_fnr))
        fconv.close()
        
        ### Create checkpoint dictionary
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.lr_scheduler.state_dict(),
            'early_stopping' : {'best_loss':early_stopping.best_loss,'counter':early_stopping.counter,'early_stop':early_stopping.early_stop,'min_delta': early_stopping.min_delta,'patience': early_stopping.patience},
            'auc': val_auc,
        }
        ### Save checkpoint
        torch.save(obj, os.path.join(args.output,'checkpoint_seed'+str(args.seed_index)+'_run'+str(args.run)+'.pth'))
        
        ### Early stopping
        if args.lr_scheduler:
            lr_scheduler(-val_auc)
        if args.early_stopping:
            early_stopping(-val_auc)
            if early_stopping.early_stop:
                break

def test(epoch, loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, (input,_) in enumerate(loader):
            ## Copy batch to GPU
            input = input.cuda()
            ## Forward pass
            y = model(input) #features, probabilities
            p = F.softmax(y,dim=1)
            ## Clone output to output vector
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = p.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(epoch, loader, model, criterion, optimizer):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, (input,target) in enumerate(loader):
        ## Copy to GPU
        input = input.cuda()
        target_1hot = F.one_hot(target.long(),num_classes=2).cuda()
        ## Forward pass
        y = model(input) #features, probabilities
        ## Calculate loss
        loss = criterion(y, target_1hot.float())
        ## Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Store loss
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

if __name__ == '__main__':
    main()
