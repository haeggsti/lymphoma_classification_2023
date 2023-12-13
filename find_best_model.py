import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='PET lymphoma classification')

parser.add_argument('--dir', type=str, default='results', help='name of folder to find convergence files in (default "results")')
parser.add_argument('--metric', type=str, default='auc', help='which metric to compare models with (default "auc")')
args = parser.parse_args()
print(args)

DF = []
files = os.listdir(args.dir)
for f in files: 
    if (f.endswith('.csv')) & (f.startswith('convergence')):
        df = pd.read_csv(os.path.join(args.dir, f))
        df = df[(df.split=='validation')&(df.metric==args.metric)].sort_values(by='epoch')
        if len(df)<7:
            print('SHORT',args.dir,f,'(',len(df),')')
            # continue
        print(args.dir,f,len(df))
        # Get last epoch
        df = df.tail(1)
        df['split'] = int( f.split('_')[1].split('split')[1] )
        df['run'] = int( f.split('run')[1].split('.')[0] )
        DF.append(df)

# Best run in this folder
DF = pd.concat(DF).reset_index(drop=True) 

grouped    = DF.groupby(['split'])
maxidx     = grouped['value'].idxmax()
dfbest     = DF.loc[maxidx]
# Top run in order of split
dfbest = dfbest.sort_values(by='split',ascending=True)
dfbest.to_csv( os.path.join(args.dir,'best_run.csv'),index=False)
print('Best run of ',args.dir,'(',str(len(dfbest)),') :',dfbest.sort_values(by='split',ascending=False).run.values)
