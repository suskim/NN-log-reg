import pandas as pd
import numpy as np
import os
import torch
import argparse
from utils import evaluation_sampler, nnlogreg_model

parser = argparse.ArgumentParser(description='att-ANN training script')
parser.add_argument('--test_df', type=str, default='', help='path to testing data csv')
parser.add_argument('--test_labels', type=str, default='', help='path to testing labels csv')
parser.add_argument('--cuda', type=str, default='0', help='which GPU to use, default: 0')
parser.add_argument('--model_path', type=str, default='', help='path to the model')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, default: 32')

def main():
    global args
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_df,index_col=0)
    test_labels = pd.read_csv(args.test_labels,index_col=0)
    
    ## dataset metrics
    print('Number of cells in the test set: ', test_labels.size, ', number of genes: ', test_df.shape[0],
        '\n    Number of clonal cells: ', np.sum(test_labels,1).item(),
        '\n    Number of normal cells: ',  test_labels.size - np.sum(test_labels,1).item())
       
    ## device configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## load the model
    curr_model = torch.load(args.model_path).to(device)
    accuracy, sensitivity, specificity, roc_auc, f1 = evaluation_sampler(curr_model, 
                                     data_file=test_df, 
                                     label_file=test_labels, 
                                     batch_size=args.batch_size,
                                     device=device) 

    print('Metrics on test set: \nAccuracy:\t',  accuracy, '\nSensitivity:\t', sensitivity, 
          '\nSpecificity:\t', specificity, '\nROC-AUC:\t', roc_auc, '\nF1:\t\t', f1)
                                
if __name__ == '__main__':
    main()
