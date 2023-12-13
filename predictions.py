import dataset
import utils
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

parser = argparse.ArgumentParser(description='PET lymphoma classification')

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output folder')
parser.add_argument('--seeds', type=int, default=None, nargs='+', help='seed range to predict')

#MODEL PARAMS
parser.add_argument('--chptfolder', type=str, default='', help='path to trained models')
parser.add_argument('--normalize', action='store_true', default=False, help='normalize images')
parser.add_argument('--nfeat', type=int, default=512, help='number of embedded features')
   
#TRAINING PARAMS
parser.add_argument('--batch_size', type=int, default=200, help='how many images to sample per slide (default: 200)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')

def main():
    # Get user input
    global args
    args = parser.parse_args()               
    # best run per seed
    bestrun = pd.read_csv(os.path.join(args.chptfolder,'best_run.csv'))
    # set per seed
    if args.seeds is not None:
        seeds = range(args.seeds[0],args.seeds[1]+1)
    else:
        seeds = range(20)
    t0 = time.time()
    for seed in seeds:
        # Best checkpoint of this seed
        run = bestrun[bestrun.seed==seed].run.values[0]
        chpnt = os.path.join(args.chptfolder,'checkpoint_seed'+str(seed)+'_run'+str(run)+'.pth')
        # Get model
        model = utils.model_prediction(chpnt)
        model.cuda()
        cudnn.benchmark = True
        # Set datasets
        _,_,_,dset,_ = dataset.get_datasets_singleview(None,args.normalize,False,seed)

        print('  dataset size: {}'.format(len(dset.df)))
        out = predict(dset, model)
        pred_name = 'pred_seed'+str(seed)+'_run'+str(run)+'.csv'
        out.to_csv(os.path.join(args.output, pred_name), index=False)
        print('  loop time: {:.0f} min'.format((time.time()-t0)/60))
        t0 = time.time()
        
def predict(dset, model):
    # Set loaders
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)    
    # Evaluate on test data
    probs, feats = test(loader, model)
    out = dset.df.copy().reset_index(drop=True)
    out['probs'] = probs
    feats = pd.DataFrame(feats, columns=['F{}'.format(x) for x in range(1,args.nfeat+1)])
    out = pd.concat([out, feats], axis=1)
    return out

def test(loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    # Initialize features
    feats = np.empty((len(loader.dataset),args.nfeat))
    # Loop through batches
    with torch.no_grad():
        for i, (input,_) in enumerate(loader):
            ## Copy batch to GPU
            input = input.cuda()
            ## Forward pass
            f, output = model.forward_feat(input)
            output = F.softmax(output, dim=1)
            ## Features
            feats[i*args.batch_size:i*args.batch_size+input.size(0),:] = f.detach().cpu().numpy()
            ## Clone output to output vector
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy(), feats

if __name__ == '__main__':
    main()
