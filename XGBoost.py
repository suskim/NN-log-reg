import pandas as pd
import numpy as np
import os
import torch
import datetime
import argparse
from utils import get_scores
import xgboost as xgb


parser = argparse.ArgumentParser(description='XGBoost')
parser.add_argument('--train_df', type=str, default='', help='path to training data csv')
parser.add_argument('--train_labels', type=str, default='', help='path to training labels csv')
parser.add_argument('--test_df', type=str, default=None, help='path to testing data csv')
parser.add_argument('--test_labels', type=str, default='', help='path to testing labels csv')
parser.add_argument('--save_dir', type=str, default='', help='where to save the model')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train, default: 1000')

def main():
    global args
    args = parser.parse_args()

    training_df = pd.read_csv(args.train_df,index_col=0)
    training_labels = pd.read_csv(args.train_labels,index_col=0)
    dtrain = xgb.DMatrix(training_df.T, label=training_labels.T)
    
    
    
    ## dataset metrics
    print('Number of cells in the training set: ', training_labels.size, ', number of genes: ', training_df.shape[0],
        '\n    Number of clonal cells: ', np.sum(training_labels,1).item(),
        '\n    Number of normal cells: ',  training_labels.size - np.sum(training_labels,1).item())
    
    if args.test_df is not None:
        test_df = pd.read_csv(args.test_df,index_col=0)
        test_labels = pd.read_csv(args.test_labels,index_col=0)
        dtest = xgb.DMatrix(test_df.T, label=test_labels.T)
        
        ## dataset metrics
        print('Number of cells in the test set: ', test_labels.size, test_df.shape[0],
            '\n    Number of clonal cells: ', np.sum(test_labels,1).item(),
            '\n    Number of normal cells: ',  test_labels.size - np.sum(test_labels,1).item())

        evallist = [(dtrain, 'train'), (dtest, 'val')]
    else:
        evallist = [(dtrain, 'train')]
    param = {'booster':'gbtree',
             'gamma':2,
             'max_delta_step':1,
             'subsample':0.8,


             'eval_metric':['rmse', 'auc']}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=20)
    if args.test_df is not None:
        preds_testset = bst.predict(dtest)
        testset = test_labels.T
        testset.columns = ['y_true']
        testset['y_out']=preds_testset
        testset_df = {}
        testset_df['y_true']=testset['y_true'].values
        testset_df['y_out']= testset['y_out'].values
        threshold=0.924
        scores_test = get_scores(testset_df, threshold)

        print('Metrics on test set: \nAccuracy:\t',  scores_test['accuracy'], '\nSensitivity:\t', scores_test['sensitivity'], 
                  '\nSpecificity:\t', scores_test['specificity'], '\nROC-AUC:\t', scores_test['auc'], '\nF1:\t\t', scores_test['f1'])

if __name__ == '__main__':
    main()
