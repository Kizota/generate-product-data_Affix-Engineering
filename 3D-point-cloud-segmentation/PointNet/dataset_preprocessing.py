'''
learning source: https://www.kaggle.com/code/jeremy26/pointnet-shapenet-dataset/notebook#Training

The testing code for classification

NUM_PARTS = 16
PART_COLORS = np.random.choice(range(255),size=(NUM_PARTS,3))

COLORS = []
for i, point in enumerate(points):
    color  = PART_COLORS[seg_labels[i] - 1]
    COLORS.append(color)

pc_plots = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],  mode='markers', marker = dict(size = 2, color = COLORS))
layout = dict(template="plotly_dark", title="Raw Point cloud", scene=PCD_SCENE, title_x=0.5) 
fig = go.Figure(data=pc_plots, layout=layout)   
fig.show()

'''
import kagglehub
import os
import sys
import json
import numpy as np
from tqdm import tqdm

# plotting library
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DL Imports
import torch
import torch.nn as nn

import glob



class ShapenetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split_type, classification = False, num_samples=2500):
        self.root_dir = root_dir + "\\Shapenetcore_benchmark"
        self.split_type = split_type
        self.classification = classification
        self.num_samples = num_samples
        with open(os.path.join(self.root_dir, f'{self.split_type}_split.json'), 'r') as f:
            self.split_data = json.load(f)       
            
    def __getitem__(self, index):
        # read point cloud data
        class_id, class_name, point_cloud_path, seg_label_path = self.split_data[index]
        
        # point cloud data
        point_cloud_path = os.path.join(self.root_dir, point_cloud_path)
        pc_data = np.load(point_cloud_path)
        
        # segmentation labels
        # -1 is to change part values from [1-16] to [0-15]
        # which helps when running segmentation
        pc_seg_labels = np.loadtxt(os.path.join(self.root_dir, seg_label_path)).astype(np.int8) - 1
#         pc_seg_labels = pc_seg_labels.reshape(pc_seg_labels.size,1)
        
        # Sample fixed number of points
        num_points = pc_data.shape[0]
        if num_points < self.num_samples:
            # Duplicate random points if the number of points is less than max_num_points
            additional_indices = np.random.choice(num_points, self.num_samples - num_points, replace=True)
            pc_data = np.concatenate((pc_data, pc_data[additional_indices]), axis=0)
            pc_seg_labels = np.concatenate((pc_seg_labels, pc_seg_labels[additional_indices]), axis=0)
                
        else:
            # Randomly sample max_num_points from the available points
            random_indices = np.random.choice(num_points, self.num_samples)
            pc_data = pc_data[random_indices]
            pc_seg_labels = pc_seg_labels[random_indices]
        
        if self.classification:
            return pc_data, class_id
        else:
            return pc_data, pc_seg_labels
        
        
        # return variable
        # data_dict= {}
        # data_dict['class_id'] = class_id
        # data_dict['class_name'] = class_name        
        # data_dict['points'] = pc_data 
        # data_dict['seg_labels'] = pc_seg_labels 
        # return data_dict        
                    
    def __len__(self):
        return len(self.split_data)



# custom imports
# from visual_utils import plot_pc_data3d, plot_bboxes_3d
#INSTALL AND PREPARE DATASET
# Download latest version
# shapenet_core_seg_path = kagglehub.dataset_download("jeremy26/shapenet-core-seg")
# print("Path to dataset files:", shapenet_core_seg_path)
# DATA_FOLDER = shapenet_core_seg_path

# train_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='train')
# val_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='val')
# test_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='test')

# print(f"Train set length = {len(train_set)}")
# print(f"Validation set length = {len(val_set)}")
# print(f"Test set length = {len(test_set)}")




