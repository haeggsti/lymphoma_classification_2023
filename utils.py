import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


def model_prediction(chpnt):
    model = my_model()
    ch = torch.load(chpnt)
    model.load_state_dict(ch['state_dict'])
    return model

def get_model():
    model = my_model()    
    model = model.cuda()
    return model

# Function that returns the model
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        # CNN
        model = models.resnet34(weights='DEFAULT')
        conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
        model._modules['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model._modules['conv1'].weight.data = conv1
        model.fc = nn.Linear(model.fc.in_features, 2)
        self.features = nn.Sequential(*list(model.children())[0:-1])
        self.fc = list(model.children())[-1]
        self.flat = True

    def forward(self, x):
        f = self.features(x)
        if self.flat:
            f = f.view(x.size(0), -1)
        o = self.fc(f)
        if not self.flat:
            o = o.view(x.size(0), -1)
        return o
    
    def forward_feat(self, x):
        f = self.features(x)
        if self.flat:
            f = f.view(x.size(0), -1)
        o = self.fc(f)
        if not self.flat:
            o = o.view(x.size(0), -1)
        return f,o

# Function that creates the optimizer
def create_optimizer(model, mode, lr, momentum, wd):
    if mode == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr,
                              momentum=momentum, dampening=0,
                              weight_decay=wd, nesterov=True)
    elif mode == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=wd)
    return optimizer

# Function to anneal learning rate
def adjust_learning_rate(optimizer, epoch, period, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every period epochs"""
    lr = start_lr * (0.1 ** (epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=1, min_lr=1e-5, factor=0.1, cooldown=1, threshold=0.001):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.cooldown = cooldown
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                cooldown=self.cooldown,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)