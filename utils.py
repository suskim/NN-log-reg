import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

## model
class nnlogreg_model(nn.Module):
    def __init__(self, repr_length : int = 44782):
        super(nnlogreg_model, self).__init__()
        
        self.repr_length = repr_length # size of representation per tile
        # N = batch size
        self.imp_matr = repr_length
        self.att = 1

        self.hypernet = nn.Sequential(             # N * repr_length
            nn.Linear(self.repr_length, 1000),    # N * D
            nn.Tanh(),                              # N * D
            nn.Linear(1000, 1000), 
            nn.Tanh(),  
            nn.Linear(1000, self.imp_matr),             # N * att
            nn.Sigmoid()               # squeeze 0-1
        )

        self.classifier = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()                              # N * repr_lenght
        A = self.hypernet(x)                      # N * att
        A = nn.functional.softmax(A, dim=1)        # softmax over att
        M = (A * x).sum(dim=1)
        out_prob = self.classifier(M.unsqueeze(-1)).view(-1)
        out_hat = torch.ge(out_prob, 0.5).float()
        
        return out_prob, out_hat, A
    
## ANN model
class ann_model(nn.Module):
    def __init__(self, repr_length : int = 44782): 
        '''
        number of features with TCR: 44949; without TCR: 44782'''
        super(ann_model, self).__init__()
        
        self.repr_length = repr_length # size of representation per tile
        # N = batch size
        self.att_matr = repr_length
        self.att = 1
        
        self.fc1 = nn.Sequential(             # N * repr_length
            nn.Linear(self.repr_length, 1000),    # N * D
            nn.SELU(),                              # N * D
            nn.Linear(1000, 1000), 
            nn.SELU(),  
            nn.Linear(1000, 1000), 
            nn.SELU(),  
            nn.Linear(1000, 32),             # N * att
            nn.SELU()               
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()                              # N * repr_lenght
        x = self.fc1(x)                      # N * att
        out_prob = self.classifier(x).view(-1)
        out_hat = torch.ge(out_prob, 0.5).float()
        A = None
        return out_prob, out_hat, A
    
    
class SCS_reader(Dataset):
    def __init__(self, data_frame, label_file, debug=False):# dset=test, 
        '''
        data_frame:    data_frame: columnsa are cells, rows are genes
        label_file:    data frame of labels (column headings are cells)
        '''
        super(SCS_reader).__init__()
        self.data_frame=data_frame
        self.labels=label_file
         
    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        name = self.labels.columns[idx]
        dx=self.labels[name].item()
        cell=self.data_frame[name].values
        return dict(x=cell, y=dx, label=name)
    

def training_sampler(model, model_name:str, save_dir:str, data_file, label_file, optimizer, batch_size, device, 
             num_epochs=5, validation = False, train_set_val=None):    
    
    criterion =  nn.BCELoss() 
    output = {}
    save_loss = []
    save_loss_val =[]
    save_acc = []
    save_acc_val = []
    y_out_train = []
    y_true_train = []
    y_out_val = []
    y_true_val = []
    train_loss = 0.
    val_loss = 0. 
    curr_val_loss=10
    curr_train_loss=10
    model_name_save=None
    
    names = label_file.columns.tolist()
    total_step = num_epochs*len(names)/batch_size
    num_samples = len(names)
    
    full_size= num_samples*num_epochs
    weight_normal = np.sum(label_file,1).item()/label_file.size
    weight_clonal = 1 - weight_normal
    
    weights_train = np.where(label_file==0, weight_normal, weight_clonal).flatten()
    
    if validation:
        ## create training and validation set
        
        n_val_train = int(np.sum(label_file,1).item())
        n_val = np.floor(n_val_train*0.2)
        n_train = n_val_train - n_val
        idx_labels_train = random.sample(range(n_val_train), int(n_train))
        idx_labels_val = [i for i in range(n_val_train) if i not in idx_labels_train]

        idx_labels_train = label_file.columns[idx_labels_train]
        idx_labels_val = label_file.columns[idx_labels_val]

        label_file_train_val = label_file[idx_labels_val]
        label_file_train_train = label_file[idx_labels_train]
        names_train_val = label_file_train_val.columns.tolist()
        names_train_train = label_file_train_train.columns.tolist()
        full_size_train_val = num_epochs * n_val
        full_size_train_train = num_epochs * n_train
        df_train_val = data_file[idx_labels_val]
        df_train_train = data_file[idx_labels_train]
        
        names = names_train_train
        data_file = df_train_train
        label_file = label_file_train_train
        
        # define weights
        weights_train = np.where(label_file==0, weight_normal, weight_clonal).flatten()
        weights_train_val = np.where(label_file_train_val==0, weight_normal, weight_clonal).flatten()
        
        # initialize validation reader
        skin_reader_val = SCS_reader(data_frame=df_train_val, label_file=label_file_train_val)
        dataloader_val = DataLoader(skin_reader_val, batch_size=batch_size, 
                                   shuffle=False, num_workers=8)

    # initiate skin readers
    skin_reader_train = SCS_reader(data_frame=data_file, label_file=label_file)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))
    dataloader_train = DataLoader(skin_reader_train, batch_size=batch_size, sampler=sampler_train, 
                                  shuffle=False, num_workers=8)
    
    for epoch in range(num_epochs):
        y_out_train.append([])
        y_true_train.append([])
        epoch_loss = 0.
        
        epoch_loss_val = 0.        
        total=0
        correct=0
        correct_val=0 
        i=0
        
        for batch in dataloader_train:
            model.train()
            
            cells = batch["x"].to(device)
            labels = batch["y"].to(device)

            y_true_train[epoch].append(labels.detach().cpu().numpy())
            i+=1
            labels = labels.float()
            total += labels.size(0)

            # Forward pass
            out_prob, out_hat, A = model.forward(cells)
            y_out_train[epoch].append(out_prob.detach().cpu().numpy())
 
            
            correct += (out_hat.float() == labels.float()).sum().item()
            loss = criterion(out_prob, labels)
       
            train_loss += loss.item()
            epoch_loss += loss.item()
            
            # reset gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # step
            optimizer.step()
            save_loss.append(epoch_loss/i)
            save_acc.append(100 * correct / total)
            
        if validation:
            y_out_val.append([])
            y_true_val.append([])
            epoch_loss_val = 0.    
            total_val=0
            correct_val = 0
            
            for batch in dataloader_val:
                model.eval()
            
                cells = batch["x"].to(device)
                labels = batch["y"].to(device)
                y_true_val[epoch].append(labels.detach().cpu().numpy())
                labels = labels.float()
                
                # Forward pass
                total_val += labels.size(0)

                # calculate loss and metrics
                out_prob, out_hat, _ = model.forward(cells)
                y_out_val[epoch].append(out_prob.detach().cpu().numpy())
                correct_val += (out_hat.float() == labels.float()).sum().item()
                loss = criterion(out_prob, labels)
                val_loss += loss.item() 
                epoch_loss_val+= loss.item()
                
                save_loss_val.append(epoch_loss_val/total_val)
                save_acc_val.append(100 * correct_val / total_val)
                
        y_out_train[epoch] = np.hstack(y_out_train[epoch])
        y_true_train[epoch] = np.hstack(y_true_train[epoch])
        y_out_val[epoch] = np.hstack(y_out_val[epoch])
        y_true_val[epoch] = np.hstack(y_true_val[epoch])
        
        if epoch%20==0:
            if validation:
                print('epoch', epoch, ':    training loss: ', (epoch_loss/total), ', validation loss: ', (epoch_loss_val/total_val))
            else: 
                print('epoch', epoch, ':    training loss: ', (epoch_loss/total))
           
        if validation:
            if epoch_loss_val/total_val < curr_val_loss:
                #print(epoch_loss_val, curr_val_loss)
                model_name_save = '{0}_{1}_epochs'.format(model_name, epoch)    
                torch.save(model, '{0}{1}.ckpt'.format(save_dir, model_name))
                curr_val_loss = epoch_loss_val/total_val
                
        else:
            if epoch_loss/total < curr_loss:
                model_name_save = '{0}_{1}_epochs'.format(model_name, epoch)    
                torch.save(model, '{0}{1}.ckpt'.format(save_dir, model_name))
                curr_loss = epoch_loss/total
                
                
    train_loss /= full_size
    print('loss: ', train_loss)
    
    # write dictionary with output
    output = {}
    output['y_out_train']=y_out_train
    output['y_true_train'] = y_true_train
    output['y_out_val'] = y_out_val
    output['y_true_val'] = y_true_val
    output['loss_train'] = save_loss
    output['loss_val'] = save_loss_val
     
    return(model, output, model_name_save)




def evaluation_sampler(model, data_file, label_file, batch_size, device):
    
    model.eval()  # eval mode 
    y_true=[]
    y_pred=[]
    y_out = []
    criterion =  nn.BCELoss() 
    
    skin_reader_test = SCS_reader(data_frame=data_file, label_file=label_file)
    dataloader_test = DataLoader(skin_reader_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.
        names = label_file.columns.tolist()
    
        num_samples= len(names)
        
        for batch in dataloader_test:
            cells = batch["x"].to(device)
            labels = batch["y"].to(device)            
            labels = labels.float()
            y_true.append(labels.detach().cpu())
                        
            total += labels.size(0)

            # forward pass: calculate loss and metrics
            out_prob, out_hat, A = model.forward(cells)
            y_out.append(out_prob.detach().cpu())
            y_pred.append(out_hat.detach().cpu())
            
            loss = criterion(out_prob, labels)
            test_loss += loss
           
           
        y_out = torch.cat(y_out).float()
        y_pred = torch.cat(y_pred).float()
        y_true = torch.cat(y_true).float()

        cm1 = confusion_matrix(y_true, y_out.round())
        sensitivity = cm1[1,1]/(cm1[1,1]+cm1[1,0])
        specificity = cm1[0,0]/(cm1[0,1]+cm1[0,0])
        fpr, tpr, _ = roc_curve(y_true, y_out)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred)
        accuracy = (cm1[0,0]+cm1[1,1]) / (cm1[1,1]+cm1[1,0]+cm1[0,0]+cm1[0,1])
        
        test_loss /= batch_size
        
        print('Loss: ', test_loss.item())
        
    return(accuracy, sensitivity, specificity, roc_auc, f1)

def Visualize(curr_output):
    y_out_train_original = curr_output['y_out_train']
    y_true_train_original = curr_output['y_true_train']
    y_out_val_original = curr_output['y_out_val']
    y_true_val_original = curr_output['y_true_val']
    val_loss = curr_output['loss_val']
    train_loss = curr_output['loss_train']
    y_out_train = y_out_train_original.copy()
    y_true_train = y_true_train_original.copy()
    y_out_val = y_out_val_original.copy()
    y_true_val = y_true_val_original.copy()


    # k is the number of epochs
    for k in range(len(y_out_train_original)):
        j = y_out_train_original[k]
        y_out_train[k] = [i for i in j]

        j = y_true_train_original[k]
        y_true_train[k] = [i for i in j]

        j = y_out_val_original[k]
        y_out_val[k] = [i.item() for i in j]

        j = y_true_val_original[k]
        y_true_val[k] = [i for i in j]
    training_y = {}
    validation_y = {}
    for i in range(len(y_out_train)):
        l = np.concatenate([[np.array(y_true_train[i])], [np.array(y_out_train[i])]])#,[np.array(y_pred)]]
        testset = pd.DataFrame(l.T, columns=['y_true', 'y_out'])#, 'y_pred'])
        testset['error'] = abs(testset.y_true - testset.y_out)
        testset['y_pred'] = np.round(testset.y_out)
        training_y[i] = testset

        l_val = np.concatenate([[np.array(y_true_val[i])], [np.array(y_out_val[i])]])#,[np.array(y_pred)]]
        testset_val = pd.DataFrame(l_val.T, columns=['y_true', 'y_out'])#, 'y_pred'])
        testset_val['error'] = abs(testset_val.y_true - testset_val.y_out)
        testset_val['y_pred'] = np.round(testset_val.y_out)
        validation_y[i] = testset_val

    auc_train = []
    acc_train = []
    f1_train = []
    sens_train = []
    spec_train = []
    auc_val = []
    acc_val = []
    f1_val = []
    sens_val = []
    spec_val = []

    for i in range(len(y_out_train)):
        testset_train = training_y[i]

        fpr, tpr, _ = roc_curve(testset_train['y_true'], testset_train['y_out'])
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(testset_train['y_true'], testset_train['y_pred'])
        cm1 = confusion_matrix(testset_train['y_true'], testset_train['y_pred'])
        correct = cm1[0,0] + cm1[1,1]
        total = testset_train.shape[0]
        accuracy = correct / total * 100
        sensitivity = cm1[1,1]/(cm1[1,1]+cm1[1,0])
        specificity = cm1[0,0]/(cm1[0,1]+cm1[0,0])
        auc_train.append(roc_auc)
        acc_train.append(accuracy)
        sens_train.append(sensitivity)
        spec_train.append(specificity)
        f1_train.append(f1)

        testset_val = validation_y[i]#print(roc_auc)
        fpr, tpr, _ = roc_curve(testset_val['y_true'], testset_val['y_out'])
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(testset_val['y_true'], testset_val['y_pred'])
        cm1 = confusion_matrix(testset_val['y_true'], testset_val['y_pred'])
        correct = cm1[0,0] + cm1[1,1]
        total = testset_val.shape[0]
        accuracy = correct / total * 100
        sensitivity = cm1[1,1]/(cm1[1,1]+cm1[1,0])
        specificity = cm1[0,0]/(cm1[0,1]+cm1[0,0])
        auc_val.append(roc_auc)
        acc_val.append(accuracy)
        sens_val.append(sensitivity)
        spec_val.append(specificity)
        f1_val.append(f1)


    ########### visulaization ##################
    epochs = len(y_out_train)
    #colors = ['r','b','g','gray','navy', 'turquoise', 'darkorange','y','k','darkgreen']
    #colors = colors[0:num_classes]

    fig, axes = plt.subplots(4,2, figsize = (10,15))

    axes[0,0].plot(range(epochs), acc_train)
    axes[0,0].set_title('TRAINING ACCURACY')

    axes[0,1].plot(range(epochs), acc_val)
    axes[0,1].set_title('VALIDATION ACCURACY')

    axes[1,0].plot(range(epochs), auc_train)
    axes[1,0].set_title('TRAINING AUC')

    axes[1,1].plot(range(epochs), auc_val)
    axes[1,1].set_title('VALIDATION AUC')

    axes[2,0].plot(range(epochs), f1_train)
    axes[2,0].set_title('TRAINING F1')

    axes[2,1].plot(range(epochs), f1_val)
    axes[2,1].set_title('VALIDATION F1')

    plt.show()

    
    
def get_scores(output, threshold=.5):
    y_out = output['y_out']
    y_true = output['y_true'] 
    y_pred = np.where(output['y_out']>threshold, 1,0)
    
    cm1 = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_out)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_true,y_pred)
    sensitivity = cm1[1,1]/(cm1[1,1]+cm1[1,0])
    specificity = cm1[0,0]/(cm1[0,1]+cm1[0,0])
    accuracy = (cm1[0,0]+cm1[1,1]) / (cm1[1,1]+cm1[1,0]+cm1[0,0]+cm1[0,1])
    
    scores = {}
    scores['accuracy']=accuracy
    scores['sensitivity']=sensitivity
    scores['specificity']=specificity
    scores['f1']=f1
    scores['auc']=roc_auc
    return scores