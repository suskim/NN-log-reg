import pandas as pd
import numpy as np
import os
import torch
import datetime
import argparse
from utils import get_scores
from pytorch_tabnet.tab_model import TabNetClassifier

parser = argparse.ArgumentParser(description='TabNet')
parser.add_argument('--train_df', type=str, default='', help='path to training data csv')
parser.add_argument('--train_labels', type=str, default='', help='path to training labels csv')
parser.add_argument('--test_df', type=str, default=None, help='path to testing data csv')
parser.add_argument('--test_labels', type=str, default='', help='path to testing labels csv')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train, default: 1000')

def main():
    global args
    args = parser.parse_args()

    training_df = pd.read_csv(args.train_df,index_col=0)
    training_labels = pd.read_csv(args.train_labels,index_col=0)
    
    
    ## dataset metrics
    print('Number of cells in the training set: ', training_labels.size, ', number of genes: ', training_df.shape[0],
        '\n    Number of clonal cells: ', np.sum(training_labels,1).item(),
        '\n    Number of normal cells: ',  training_labels.size - np.sum(training_labels,1).item())
    
    if args.test_df is not None:
        test_df = pd.read_csv(args.test_df,index_col=0)
        test_labels = pd.read_csv(args.test_labels,index_col=0)
        
        ## dataset metrics
        print('Number of cells in the test set: ', test_labels.size, test_df.shape[0],
            '\n    Number of clonal cells: ', np.sum(test_labels,1).item(),
            '\n    Number of normal cells: ',  test_labels.size - np.sum(test_labels,1).item())
        eval_set=[(np.array(test_df.T), np.array(test_labels)[0])]
    else:
        eval_set=None

    clf = TabNetClassifier()
    clf.fit(
        np.array(training_df.T), 
        np.array(training_labels)[0],
        eval_set=eval_set,
        max_epochs=args.num_epochs,
        eval_metric=['balanced_accuracy', 'auc'],
        patience=100)
    
    if args.test_df is not None:
        preds_testset = clf.predict_proba(np.array(test_df.T))
        testset = test_labels.T
        testset.columns = ['y_true']
        testset['y_out']=preds_testset[:,1]
        testset_df = {}
        testset_df['y_true']=testset['y_true'].values
        testset_df['y_out']= testset['y_out'].values
        threshold=0.735
        scores_test = get_scores(testset_df, threshold)

        print('Metrics on test set: \nAccuracy:\t',  scores_test['accuracy'], '\nSensitivity:\t', scores_test['sensitivity'], 
                  '\nSpecificity:\t', scores_test['specificity'], '\nROC-AUC:\t', scores_test['auc'], '\nF1:\t\t', scores_test['f1'])

if __name__ == '__main__':
    main()
