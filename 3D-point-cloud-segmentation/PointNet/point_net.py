import torch 
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    '''T-Net learns a Transformation matrix with a specified dimension, the result is an 3D rotation matrix'''
    '''T_net has the purpose of convert every data point to the same cosistent orientation'''
    '''To answer the question how could be the final 3x3 rotation is the roation matrix is it about supervised leanring'''
    def __init__(self, dim, num_points = 2500):
        super(Tnet, self).__init__()

        #dimensions for transform matrix - what is this?
        self.dim = dim

        #MLPs (1d convolutional layers) for 3D - extract the features to claculate the conssitent orientation
        self.conv1 = nn.Conv1d(dim, 64, kernel_size = 1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 1)
        self.conv3 = nn.Conv1d(128,1024, kernel_size = 1)

        #three fully connected layers (152, 256, 9) 
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,dim**2) 

        #batch norm layers
        '''
        simply put, the batch norm layers recentered the output data of a layer around 0, so the model can learned more features representation with activation function
         better understanding of batch norm, why and how is bath norm work.
        https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
        
        '''  
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        #Max poolng 1024
        self.max_pool = nn.MaxPool1d(kernel_size = num_points)


    def forward(self, x):
        #record the bath size, the number of given n
        # print("in TNET ---------")

        # print(x.shape[0])
        bs = x.shape[0] 
         
        x =  self.bn1(F.relu(self.conv1(x)))
        x =  self.bn2(F.relu(self.conv2(x)))
        x =  self.bn3(F.relu(self.conv3(x)))

        #max pool over num points
        x = self.max_pool(x).view(bs,-1)

        #pass through Fully connected layers 
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)
      
        #reshape to the rotation matrix - for stability
        iden = torch.eye(self.dim, requires_grad =True).repeat(bs,1,1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view( -1, self.dim, self.dim) + iden
        
        return x

class PointNetBackbone(nn.Module):
    '''
    The basic principle of PointNet is about to extract the local and global feature.
    '''
    def __init__(self,num_points = 2500, num_global_feats = 1024, local_feat = True):
        super(PointNetBackbone,self).__init__()
        
        self.num_points = num_points
        self.num_global_feats = num_global_feats 
        self.local_feat = local_feat

        # 2 Tnet for rotating the the point cloud to a consistent orientation.
        self.tnet1 = Tnet(3) #3 features (3D coordinates)
        self.tnet2 = Tnet(64) #64 features  (extracted abstract features)

        #MLP 1 with 2 1D convolution layers (64,64)
        self.conv1 =  nn.Conv1d(3,64,kernel_size = 1)
        self.conv2 =  nn.Conv1d(64,64,kernel_size = 1)      
       
       #bath norm for MLPS and Tnet 
        self.conv3 = nn.Conv1d(64,64,kernel_size = 1)
        self.conv4 = nn.Conv1d(64,128,kernel_size = 1)
        self.conv5 = nn.Conv1d(128,self.num_global_feats,kernel_size = 1)

        #bath norm layers for relearn more inside representation in the point cloud
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        #1 max pooling for extracting the global feature
        self.max_pool = nn.MaxPool1d(kernel_size = num_points,return_indices=True)
        
        
    def forward(self,x):
        #get the btch size
        bs = x.shape[0]
        # print("BEGIN OF FIRST MSL BACKBONE ---------")
        # print(x.shape)
        
    
        #pass through first Tnet to get transform matrix 
        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
       
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        
        # print("BEGIN OF THE SECOND MSL BACKBONE  ---------")
        # print(x.shape)
        #get the feature transform 
        A_feat = self.tnet2(x)

        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        
        # store local point features for segmentation head
        local_features = x.clone()

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # get global feature vector and critical indexes
        '''
        The global features tell what kind of the abstract features.
        The cirical_indexes tell in which data point hole the max values of these features.
        The global featues is an 2x2 matrix of data points and features' values. 
        Meaning it still holes the features information of all datapoints.
        '''

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)
       
        #co
        if self.local_feat:
            features = torch.cat((local_features, 
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                                  dim=1)

            return features, critical_indexes, A_feat

        else:
            return global_features, critical_indexes, A_feat 
    
#Classification head
class PointNetClassHead(nn.Module):
    '''
    Initialising parameters/ model hyper parameters:
     + num_points: the points numbers of the input point clouds (n)
     + num_global_feats: the number of global features
     + the number of possible classification outputs.
     
    Question: 
     1. why the cirical set points is not used?
    '''
    def __init__(self, num_points = 2500, num_global_feats = 1024, k = 2): 
        super(PointNetClassHead,self).__init__()

        # Get the backbone - only need the global features for classificcation
        self.backbone = PointNetBackbone(num_points,num_global_feats,False)

        #MLP with 3 1d fully connected  layers (512, 256, k)
        self.linear1 = nn.Linear(num_global_feats,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,k)

        #batch norm layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        #prevent the overfitting phenomenon by reducing the noise data/ randomly blocking a certain number of neurons (30% in this case)
        #the dropout layer is applied to hidden layer before the input layer
        '''never apply the dropout layer to the output layer, no meaning, since the output can not be blocked'''
        self.dropout = nn.Dropout(0.3)
        

    def forward(self,x):
        #get global features
        x, crit_idxs, A_feat = self.backbone(x)

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        #return logits 
        return x, crit_idxs, A_feat
        

#segmentation head 
class PointNetSegHead(nn.Module):
    '''
    1. why there are only 4 conv layers? following the diagrams, there are five in total.
    '''
    def __init__(self,num_points = 2500, num_global_feats = 1024, m = 2): 
        super(PointNetSegHead,self)
        
        self.num_points = num_points
        self.m = m

        #Get both the local and global features for segmentation withPointNet backbone.
        self.backbone = PointNetBackbone( num_points,num_global_feats, True)

        #first shared MLP with 3 1d convolutional layers 
        self.conv1 = nn.Conv1d(1088,512,kernel_size = 1)
        self.conv2 = nn.Conv1d(512,256,kernel_size = 1)
        self.conv3 = nn.Conv1d(256,128,kernel_size = 1)

       #second shared MLP with 2 1d convolutional layers
        self.conv4 = nn.Conv1d(128,128, kernel_size = 1)
        self.conv5 = nn.Conv1d(128,m,kernel_size = 1)

        #batch norm layers for each of the convl layer 
        self.bn1 = nn.BatchNorm(512)
        self.bn2 = nn.BatchNorm(256)
        self.bn3 = nn.BatchNorm(128)
        self.bn4 = nn.batchNorm(128)

    
    def forward(self,x):
        #get both local and global features 
        x, crit_idxs, A_feat = self.backbone(x)
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))       
        x = self.conv5(x)   

        x = x.transpose(2,1)
        return x, crit_idxs, A_feat
