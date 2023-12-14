import numpy as np
import pandas as pd
import os
import glob
import torch
import torch.nn as nn
import random
import itertools
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from scipy import interpolate

data_path_img = '<path-to-binary-image-files>'
data_file     = '<path-and_name-of-csv-datafile>'

def get_datasets_singleview(transform=None, norm=None, balance=False, split_index=0):
    split = 'split'+str(split_index)
    df = pd.read_csv(data_file)
    # Balance weight
    weight_neg_pos = [1-(df.target==0).sum()/len(df), 1-(df.target==1).sum()/len(df)]    
    # Read split
    df_train = df[df[split]=='train'].drop(df.filter(regex='split').columns,axis=1)
    train_dset = dataset_singleview(df_train, transform=transform, norm=norm)
    trainval_dset = dataset_singleview_center(df_train, transform=None, norm=norm)
    # Val split
    df_val = df[df[split]=='val'].drop(df.filter(regex='split').columns,axis=1)
    val_dset = dataset_singleview_center(df_val, transform=None, norm=norm)
    # Test split
    df_test = df[df[split]=='test'].drop(df.filter(regex='split').columns,axis=1)
    test_dset = dataset_singleview_center(df_test, transform=None, norm=norm)
    return train_dset,trainval_dset,val_dset,test_dset,weight_neg_pos

def get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax, cmin:cmax]

def pad2square_random(image, size):
    out = np.zeros((size,size))
    # Sample offset
    maxr = size - image.shape[0]
    maxc = size - image.shape[1]
    offsetc = np.random.randint(0, maxc)
    offsetr = np.random.randint(0, maxr)
    # Place image
    out[offsetr:offsetr+image.shape[0], offsetc:offsetc+image.shape[1]] = image
    return out

def pad2square_center(image, size):
    # Place image
    out = np.zeros((size,size))
    out[int((size-image.shape[0])/2):int((size-image.shape[0])/2)+image.shape[0],int((size-image.shape[1])/2):int((size-image.shape[1])/2)+image.shape[1]] = image
    return out

def clip_and_normalize_SUVimage(img):
    mu  = 2.13
    std = 3.39
    q   = 30.00
    img = np.clip(img,0.,q)
    return (img-mu)/std

def get_image(df, transform, norm):
    image_size = 310
    name = glob.glob(os.path.join(data_path_img,df.filename))
    if not name:
        print('File not found:',name)
    img  = np.fromfile(os.path.join(data_path_img, name[0]), dtype='float32')
    img  = np.reshape( img,[df.matrix_size_1, df.matrix_size_2])
    # Find bbox
    img = get_bbox(img)
    # Pad randomly
    img = pad2square_random(img, image_size)
    # Norm
    if norm:
        img = clip_and_normalize_SUVimage(img)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img
    
def get_image_center(df, transform, norm):
    image_size = 310
    name = glob.glob(os.path.join(data_path_img,df.filename))
    if not name:
        print('File not found:',name)
    img  = np.fromfile(os.path.join(data_path_img, name[0]), dtype='float32')
    img  = np.reshape( img,[df.matrix_size_1, df.matrix_size_2])
    # Find bbox
    img = get_bbox(img)
    # Pad randomly
    img = pad2square_center(img, image_size)
    # Norm
    if norm:
        img = clip_and_normalize_SUVimage(img)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img

class dataset_singleview(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, norm=False):
        self.df = df.copy()
        self.transform = transform
        self.norm = norm
    
    def errors(self, probs):
        df = self.df.copy()
        df['p'] = probs
        df['pred'] = (df.p >= 0.5).astype(int)
        fpr = ((df.pred!=df.target) & (df.target==0)).sum() / (df.target==0).sum()
        fnr = ((df.pred!=df.target) & (df.target==1)).sum() / (df.target==1).sum()
        ber = (fpr + fnr) / 2.
        ## Calculate auc
        auc = roc_auc_score(df.target, df.p)
        return auc, ber, fpr, fnr
    
    def __getitem__(self, index): 
        df = self.df.iloc[index]
        # Read image
        img = get_image(df, self.transform, self.norm)
        return img, df.target
    
    def __len__(self):
        return len(self.df)

class dataset_singleview_center(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, norm=False):
        self.df = df.copy()
        self.transform = transform
        self.norm = norm
    
    def errors(self, probs):
        df = self.df.copy()
        df['p'] = probs
        df['pred'] = (df.p >= 0.5).astype(int)
        fpr = ((df.pred!=df.target) & (df.target==0)).sum() / (df.target==0).sum()
        fnr = ((df.pred!=df.target) & (df.target==1)).sum() / (df.target==1).sum()
        ber = (fpr + fnr) / 2.
        ## Calculate auc
        auc = roc_auc_score(df.target, df.p)
        return auc, ber, fpr, fnr
    
    def __getitem__(self, index): 
        df = self.df.iloc[index]
        # Read image
        img = get_image_center(df, self.transform, self.norm)
        return img, df.target
    
    def __len__(self):
        return len(self.df)

class RandomFlip(object):
    """Randomly flip the 2D image.
    """
    def __call__(self, image):
        # Random flip: none, 0=vertical, 1=horizontal
        flip = random.choice((None,0,1))
        if flip is not None:
            if flip==0:
                image   = image[range(image.shape[flip]-1,-1,-1),:]
            elif flip==1:
                image   = image[:,range(image.shape[flip]-1,-1,-1)]       
        return image

class RandomFlipLeftRight(object):
    """Randomly flip all channels of the 2D image.
    """
    def __call__(self, image):
        # Random flip: none, 0=vertical, 1=horizontal
        flip = random.choice((None,1))
        if flip is not None:
            image = image[:,range(image.shape[1]-1,-1,-1)]
        return image
        
class RandomRot90(object):
    """Randomly rotate the 2D image by n*90 degrees.
    """
    def __call__(self, image):
        # Random  90 rotation
        rot = random.randint(0,3)
        if rot != 0:
            image = torch.rot90(image, rot, (0,1))
        return image

class RandomScale(object):
    """Randomly scale the 2D image.
    """
    def __call__(self, image):
        scale = np.random.uniform(low=0.85, high=1.15, size=1)
        image = image*scale[0]
        return image

class RandomNoise(object):
    """Randomly gauss noise the 2D image.
    """
    def __call__(self, image):
        noise = random.choice((None,1))
        if noise is not None:
            image[image<0] = 0
            level = np.random.uniform(low=0.001, high=0.02, size=1)
            sigma = np.random.uniform(low=0.01, high=0.1, size=1)
            sigma = sigma[0]*image+level[0]
            gauss = torch.normal(0,sigma)
            image = image + gauss
            image[image<0] = 0
        return image