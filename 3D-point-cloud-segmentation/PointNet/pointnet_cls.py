import os 
import json
import numpy as np
import time

#import deep learning train modules 
from point_net import PointNetClassHead
from point_net_los import PointNetLoss
import torch.optim as optim
import torch
from torchmetrics.classification import MulticlassMatthewsCorrCoef

#import data modules 
import kagglehub 
from torch.utils.data import DataLoader
from dataset_preprocessing import ShapenetDataset

# #GET DATA 
BATCH_SIZE =32
NUM_CLASSES = 16

#get class - label mappings 
CATEGORIES = {
    'Airplane': 0, 
    'Bag': 1, 
    'Cap': 2, 
    'Car': 3,
    'Chair': 4, 
    'Earphone': 5, 
    'Guitar': 6, 
    'Knife': 7, 
    'Lamp': 8, 
    'Laptop': 9,
    'Motorbike': 10, 
    'Mug': 11, 
    'Pistol': 12, 
    'Rocket': 13, 
    'Skateboard': 14, 
    'Table': 15}

#installing and splitting up dataset 
ROOT = kagglehub.dataset_download("jeremy26/shapenet-core-seg")

'''
the dataset manage the data in its set and give an object data based on the requirmenet 
the dataloader is an data feeding algorithm to the model. It instructs the feed process in batch and go through many time.
With this shapenetDataset implemetnation, the data is splitted based on the json file configuration not by manually
'''
#train dataset  
train_dataset = ShapenetDataset(root_dir = ROOT, classification= True, split_type = 'train')   #the dataset object holds the rerfence and configuration of the dataset
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle = True)   #the dataloader is an array of feeding data batches to the model 

valid_dataset = ShapenetDataset(root_dir = ROOT, classification = True, split_type = 'val')
valid_dataloader = DataLoader(valid_dataset,batch_size = BATCH_SIZE)

test_dataset = ShapenetDataset(root_dir = ROOT, classification = True, split_type = 'test')
test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE)


#TRAIN THE MODEL 
#general parameters 
EPOCHS = 100         #epoch is the number of time that the dataset is feeded  to the model for training 
LR = 0.0001         #lr is the learning rate, representing how much should the gardient is decreased each of everytime to reach the lowest point.
REG_WEIGHT = 0.001  #the requglation wieght prevent overfitting and improve general performance by penaliszing large weight. This is used for the loss function.

#model parameters 
GLOBAL_FEATS = 1024

'''
question: what is aplha ???????
'''
alpha = np.ones(NUM_CLASSES)
alpha[0] = 0.5 
alpha[4] = 0.5
alpha[-1] = 0.5

gamma = 2

DEVICE = 'cpu'

#initialise train functions and objects. (pointnet classifier, loss funcition, optimizer)
classifier = PointNetClassHead(k = NUM_CLASSES, num_global_feats = GLOBAL_FEATS)
optimizer = optim.Adam(classifier.parameters(), lr= LR)
criterion = PointNetLoss(alpha = alpha, gamma = gamma, reg_weight = REG_WEIGHT).to(DEVICE)

#Matthews
mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)

def train_test(classifier, dataloader, num_batch, epoch, split = 'train'):
    ''' Function to train or test the model'''
    _loss = []
    _accuracy = []   #testing 
    _mcc = [] 
    
    #return total targets and predictions for test case only
    total_test_targets = []
    total_test_preds = []
    '''
    Question:
    1. why transpose the points here, what is the data structure of the poitns?
    '''
    for i,(points, targets) in enumerate(dataloader, 0): #points here mean a array of different point cloud
    
        points = points.transpose(2,1).to(DEVICE)
        targets = targets.squeeze().to(DEVICE)
 
        #zero gradients - need to set back to zero as pytorch accumulate the gardient by default.
        optimizer.zero_grad() 
     
        #forward - get the prediction - the classifier only predict one 
        preds, _, A = classifier(points)
           
        #get loss values  - this loss is an object 
        loss = criterion(preds, targets, A)
        
        #backward - updating the weigths parameters 
        if split == 'train':
            loss.backward()   #this would calculate the loss for each of every parameters storing them in a seperate places in the classifier object 
            optimizer.step()  #this would per
        
        #get class prediction 
        pred_choice = torch.softmax(preds, dim = 1).argmax(dim = 1)
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct.item()/float(BATCH_SIZE)
        mcc = mcc_metric(preds, targets)
        
        # update epoch loss and accuracy
        _loss.append(loss.item())
        _accuracy.append(accuracy)
        _mcc.append(mcc.item())

        
        # add to total targets/preds 
        if split == 'test':
            total_test_targets += targets.reshape(-1).cpu().numpy().tolist()
            total_test_preds += pred_choice.reshape(-1).cpu().numpy().tolist()
    
        if i % 100 != 0:
                print(f'\t [{epoch}: {i}/{num_batch}] ' \
                    + f'{split} loss: {loss.item():.4f} ' \
                    + f'accuracy: {accuracy:.4f} mcc: {mcc:.4f}')
            
    epoch_loss = np.mean(_loss)
    epoch_accuracy = np.mean(_accuracy)
    epoch_mcc = np.mean(_mcc)

    print(f'Epoch: {epoch} - {split} Loss: {epoch_loss:.4f} ' \
            + f'- {split} Accuracy: {epoch_accuracy:.4f} ' \
            + f'- {split} MCC: {epoch_mcc:.4f}')

    if split == 'test':
            return epoch_loss, epoch_accuracy, epoch_mcc, total_test_targets, total_test_preds
    else: 
            return epoch_loss, epoch_accuracy, epoch_mcc
    
    
# stuff for training
num_train_batch = int(np.ceil(len(train_dataset)/BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset)/BATCH_SIZE))

    
# store best validation mcc above 0.
best_mcc = 0.

# lists to store metrics (loss, accuracy, mcc)
train_metrics = []
valid_metrics = []

# TRAIN ON EPOCHS
for epoch in range(1, EPOCHS):
    ## train loop
    classifier = classifier.train()
    
    # train
    _train_metrics = train_test(classifier, train_dataloader, 
                                num_train_batch, epoch, 
                                split='train')
    train_metrics.append(_train_metrics)
    
    # pause to cool down
    time.sleep(4)






