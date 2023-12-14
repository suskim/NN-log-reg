import pandas as pd
import numpy as np
import os
import torch
import datetime
import argparse
from utils import Visualize, training_sampler, ann_model, evaluation_sampler

now=datetime.datetime.now()
parser = argparse.ArgumentParser(description='att-ANN training script')
parser.add_argument('--train_df', type=str, default='', help='path to training data csv')
parser.add_argument('--train_labels', type=str, default='', help='path to training labels csv')
parser.add_argument('--test_df', type=str, default=None, help='path to testing data csv')
parser.add_argument('--test_labels', type=str, default='', help='path to testing labels csv')
parser.add_argument('--cuda', type=str, default='0', help='which GPU to use, default: 0')
parser.add_argument('--save_dir', type=str, default='', help='where to save the model')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train, default: 1000')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for ANN training, default: 0.0001')
parser.add_argument('--eps', type=float, default=1E-10, help='Epsilon parameter for RMSprop optimizer, default: 01E-10')
parser.add_argument('--weightdecay', type=float, default=0.1, help='Weight decay parameter for RMSprorp optimizer, default: 0.1')
parser.add_argument('--momentum', type=float, default=0.5, help='Momentum parameter for RMSprop optimizer, default: 0.5')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, default: 32')
parser.add_argument('--alpha', type=float, default=0.99, help='Alpha parameter for RMSprop optimizer, default: 0.99')
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
    curr_model = ann_model(num_features).to(device)
    optimizer = torch.optim.RMSprop(curr_model.parameters(), 
                                    lr=args.lr, 
                                    alpha=args.alpha, 
                                    eps=args.eps,
                                    weight_decay=args.weightdecay,
                                    momentum=args.momentum)
    
    now=datetime.datetime.now()
    model_name = f'{args.date}_annmodel_RMSprop_lr{args.lr}_eps{args.eps}_alpha{args.alpha}_weightdecay{args.weightdecay}_momentum{args.momentum}'
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
    
    if args.test_df is not None:
        test_df = pd.read_csv(args.test_df,index_col=0)
        test_labels = pd.read_csv(args.test_labels,index_col=0)

        ## dataset metrics
        print('Number of cells in the test set: ', test_labels.size, test_df.shape[0],
            '\n    Number of clonal cells: ', np.sum(test_labels,1).item(),
            '\n    Number of normal cells: ',  test_labels.size - np.sum(test_labels,1).item())

        ## device configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        accuracy, sensitivity, specificity, roc_auc, f1 = evaluation_sampler(curr_model, 
                                         data_file=test_df, 
                                         label_file=test_labels, 
                                         batch_size=args.batch_size,
                                         device=device) 

        print('Metrics on test set: \nAccuracy:\t',  accuracy, '\nSensitivity:\t', sensitivity, 
              '\nSpecificity:\t', specificity, '\nROC-AUC:\t', roc_auc, '\nF1:\t\t', f1)


if __name__ == '__main__':
    main()
