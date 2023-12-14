import pandas as pd
import numpy as np
import os
import torch
import datetime
import argparse
from utils import Visualize, training_sampler, nnlogreg_model

now=datetime.datetime.now()
parser = argparse.ArgumentParser(description='att-ANN training script')
parser.add_argument('--train_df', type=str, default='', help='path to training data csv')
parser.add_argument('--train_labels', type=str, default='', help='path to training labels csv')
parser.add_argument('--cuda', type=str, default='0', help='which GPU to use, default: 0')
parser.add_argument('--save_dir', type=str, default='', help='where to save the model')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train, default: 1000')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for ANN training, default: 0.0001')
parser.add_argument('--eps', type=float, default=1E-08, help='Epsilon parameter for RMSprop optimizer, default: 01E-08')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, default: 32')
parser.add_argument('--alpha', type=float, default=0.90, help='Alpha parameter for RMSprop optimizer, default: 0.90')
parser.add_argument('--date', type=str, default=now.strftime("%m-%d-%Y"), help='date to use for saving the model, default: today')

def main():
    global args
    args = parser.parse_args()

    training_df = pd.read_csv(args.train_df,index_col=0)
    training_labels = pd.read_csv(args.train_labels,index_col=0)
    
    ## dataset metrics
    print('Number of cells in the training set: ', training_labels.size, ', number of genes: ', training_df.shape[0],
        '\n    Number of clonal cells: ', np.sum(training_labels,1).item(),
        '\n    Number of normal cells: ',  training_labels.size - np.sum(training_labels,1).item())
       
    ## device configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## train the model
    num_features = training_df.shape[0]
    curr_model = nnlogreg_model(num_features).to(device)
    optimizer = torch.optim.RMSprop(curr_model.parameters(), 
                                    lr=args.lr, 
                                    alpha=args.alpha, 
                                    eps=args.eps,
                                    weight_decay=0.0,
                                    momentum=0)
    
    
    if args.save_dir != '':
        os.chdir(args.save_dir)
    now=datetime.datetime.now()
    model_name = f'{args.date}_model_RMSprop_lr{args.lr}_eps{args.eps}_alpha{args.alpha}'
    curr_model, curr_output, model_name_save = training_sampler(curr_model, 
                                                                model_name=model_name, 
                                                                save_dir=args.save_dir,
                                                                data_file=training_df, 
                                                                label_file=training_labels,
                                                                optimizer=optimizer, 
                                                                batch_size=args.batch_size, 
                                                                num_epochs=args.num_epochs, 
                                                                validation = True, 
                                                                train_set_val=None, 
                                                                device=device)  
   
    print(model_name_save)
    Visualize(curr_output)
    print('Time = ', datetime.datetime.now()-now)
                                
if __name__ == '__main__':
    main()
