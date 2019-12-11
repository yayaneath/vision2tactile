from os import listdir

import numpy as np
import pandas as pd
from pypcd import pypcd

import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, pcds_folder, cloud_type, tgt_cols, tgt_norm_val, transform=None):
        self.pcds_folder = pcds_folder
        self.cloud_type = cloud_type
        self.folders = listdir(pcds_folder)
        self.transform = transform
        self.target_cols = tgt_cols
        self.target_norm_val = tgt_norm_val

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        sample_id = self.folders[idx]

        pcd_path = self.pcds_folder + '/' + sample_id + '/cloud-' + self.cloud_type + '.pcd'
        point_cloud = pypcd.PointCloud.from_path(pcd_path)
        cloud = point_cloud.pc_data.view(np.float32).reshape(point_cloud.pc_data.shape + (-1,))
        cloud = np.delete(cloud, [3], axis=1) # TODO: Parametrise colour removing

        tac_path = self.pcds_folder + '/' + sample_id + '/tactile.tac'
        tac_pd = pd.read_csv(tac_path, header=None, usecols=self.target_cols)

        if len(self.target_cols) > 1:
            target = tac_pd.values[0] / self.target_norm_val
        else:
            target = tac_pd.values[0][0] / self.target_norm_val

        if self.transform:
            cloud = self.transform(cloud)

        return cloud, target