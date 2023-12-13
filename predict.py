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
parser.add_argument('--output', type=str, default='.', help='name of output folder (default: ".")')
parser.add_argument('--splits', type=int, default=None, nargs='+', help='data split range to predict for')

#MODEL PARAMS
parser.add_argument('--chptfolder', type=str, default='results', help='path to trained models (default: "results")')
parser.add_argument('--normalize', action='store_true', default=False, help='normalize images')
parser.add_argument('--nfeat', type=int, default=512, help='number of embedded features (default: 512)')
   
#TRAINING PARAMS
parser.add_argument('--batch_size', type=int, default=200, help='how many images to sample per slide (default: 200)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')

def main():
    # Get user input
    global args
    args = parser.parse_args()               
    # best run per split
    bestrun = pd.read_csv(os.path.join(args.chptfolder,'best_run.csv'))
    # set per split
    if args.splits is not None:
        splits = range(args.splits[0],args.splits[1]+1)
    else:
        splits = range(20)
    t0 = time.time()
    for split in splits:
        # Best checkpoint of this split
        run = bestrun[bestrun.split==split].run.values[0]
        chpnt = os.path.join(args.chptfolder,'checkpoint_split'+str(split)+'_run'+str(run)+'.pth')
        # Get model
        model = utils.model_prediction(chpnt)
        model.cuda()
        cudnn.benchmark = True
        # Test dataset
        _,_,_,dset,_ = dataset.get_datasets_singleview(None,args.normalize,False,split)

        print('  dataset size: {}'.format(len(dset.df)))
        out = predict(dset, model)
        pred_name = 'pred_split'+str(split)+'_run'+str(run)+'.csv'
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
