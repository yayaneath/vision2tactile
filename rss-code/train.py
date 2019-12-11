import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

from sklearn.model_selection import ShuffleSplit

import argparse
import numpy as np
import matplotlib.pyplot as plt

from model.pointnet import PointNet
from dataset.pointcloud import PointCloudDataset
from utils.rmseloss import RMSELoss
from transforms import *


MF = 'MF'
TH = 'TH'
TGT_PDC = 'PDC'
TGT_ELEC = 'ELEC'
CLOUD_TYPE = 'object'
TRAIN_PATH = './data/train'
TEST_PATH = './data/test'


# These normalisation values were obtained empirically
# (They stand as the maximum registered values for each finger)

def get_task_params(finger, target):
    if finger == MF and target == TGT_PDC:
        target_cols = [27]
        target_norm_value = 2000
    elif finger == MF and target == TGT_ELEC:
        target_cols = list(range(1, 25))
        target_norm_value = 3800
    elif finger == TH and target == TGT_PDC:
        target_cols = [56]
        target_norm_value = 3100
    elif finger == TH and target == TGT_ELEC:
        target_cols = list(range(30, 53))
        target_norm_value = 3500

    return target_cols, target_norm_value

def train_model(tgt_cols, tgt_norm_val, train_params):
    train_folder = TRAIN_PATH
    output_vars = len(tgt_cols)
    device = train_params['device']

    cloud_dataset = PointCloudDataset(pcds_folder=train_folder, cloud_type=CLOUD_TYPE,
                                      tgt_cols=tgt_cols, tgt_norm_val=tgt_norm_val,
                                      transform=transforms.Compose([
                                           downsample.Downsample(train_params['downsample_size']),
                                           normalise.NormaliseUnitSphere(),
                                           totensor.ToTensor()
                                       ]))

    train_dataloader = DataLoader(cloud_dataset, batch_size=train_params['batch_size'],
                                 sampler=RandomSampler(cloud_dataset),
                                 num_workers=train_params['num_workers'])

    pointnet_model = PointNet(input_channels=train_params['cloud_fields'],
                              output_vars=output_vars).to(device)
    
    criterion = RMSELoss()
    optimiser = torch.optim.SGD(pointnet_model.parameters(), lr=train_params['learning_rate'])

    losses = []

    for epoch in range(train_params['epochs']):
        epoch_loss = []

        for i_batch, sample_batch  in enumerate(train_dataloader):
            cloud_batch, label_batch = sample_batch
            label_batch = label_batch.view(label_batch.shape[0], -1).float()
            cloud_batch, label_batch = cloud_batch.to(device), label_batch.to(device)

            optimiser.zero_grad()

            # Input must be [batch, channels (i.e. x,y,z), points]
            output = pointnet_model(cloud_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())

        losses.append(np.mean(epoch_loss))

        if (epoch + 1) % 5 == 0:
            print('Epoch', epoch + 1, 'loss:', losses[-1], '> scaled:', losses[-1] * tgt_norm_val)

    #plt.plot(losses)
    #plt.title('Training loss')
    #plt.show()

    return pointnet_model, losses

def test_model(tgt_cols, tgt_norm_val, params, model):
    test_folder = TEST_PATH
    device = params['device']

    cloud_dataset = PointCloudDataset(pcds_folder=test_folder, cloud_type=CLOUD_TYPE,
                                      tgt_cols=tgt_cols, tgt_norm_val=tgt_norm_val,
                                      transform=transforms.Compose([
                                           downsample.Downsample(params['downsample_size']),
                                           normalise.NormaliseUnitSphere(),
                                           totensor.ToTensor()
                                       ]))

    test_dataloader = DataLoader(cloud_dataset, batch_size=params['batch_size'],
                                 num_workers=params['num_workers'])
    
    criterion = RMSELoss()

    with torch.no_grad():
        test_losses = []

        for sample_batch in test_dataloader:
            cloud_batch, label_batch = sample_batch
            label_batch = label_batch.view(label_batch.shape[0], -1).float()
            cloud_batch, label_batch = cloud_batch.to(device), label_batch.to(device)

            output = model(cloud_batch)
            loss = criterion(output, label_batch)
            test_losses.append(loss.item())
    
        print('>> TEST LOSS:', np.mean(test_losses), '+/-', np.std(test_losses))
        print('>> TEST LOSS (SCALED):', np.mean(test_losses) * tgt_norm_val, 
            '+/-', np.std(test_losses) * tgt_norm_val)

    return test_losses

def train_model_cv(tgt_cols, tgt_norm_val, train_params):
    folds = 5
    train_folder = TRAIN_PATH
    output_vars = len(tgt_cols)
    device = train_params['device']

    cloud_dataset = PointCloudDataset(pcds_folder=train_folder, cloud_type=CLOUD_TYPE,
                                      tgt_cols=tgt_cols, tgt_norm_val=tgt_norm_val,
                                      transform=transforms.Compose([
                                           downsample.Downsample(train_params['downsample_size']),
                                           normalise.NormaliseUnitSphere(),
                                           totensor.ToTensor()
                                       ]))

    samples = len(cloud_dataset)
    kf = ShuffleSplit(n_splits=folds, test_size=samples/(folds*100), random_state=0)
    fold_count = 1
    folds_losses = []
    
    for train_idx, test_idx in kf.split(range(samples)):
        print('\n>> TRAINING FOLD', fold_count)

        train_dataloader = DataLoader(cloud_dataset, batch_size=train_params['batch_size'],
                                     sampler=SubsetRandomSampler(train_idx),
                                     num_workers=train_params['num_workers'])

        pointnet_model = PointNet(input_channels=train_params['cloud_fields'],
                                  output_vars=output_vars).to(device)
        
        criterion = RMSELoss()
        optimiser = torch.optim.SGD(pointnet_model.parameters(), lr=train_params['learning_rate'])

        losses = []

        for epoch in range(train_params['epochs']):
            epoch_loss = []

            for i_batch, sample_batch  in enumerate(train_dataloader):
                cloud_batch, label_batch = sample_batch
                label_batch = label_batch.view(label_batch.shape[0], -1).float()
                cloud_batch, label_batch = cloud_batch.to(device), label_batch.to(device)

                optimiser.zero_grad()

                # Input must be [batch, channels (i.e. x,y,z), points]
                output = pointnet_model(cloud_batch)
                loss = criterion(output, label_batch)
                loss.backward()
                optimiser.step()

                epoch_loss.append(loss.item())

            losses.append(np.mean(epoch_loss))

            if (epoch + 1) % 5 == 0:
                print('Epoch', epoch + 1, 'loss:', losses[-1], '> scaled:', losses[-1] * tgt_norm_val)

        test_dataloader = DataLoader(cloud_dataset, batch_size=train_params['batch_size'],
                                     sampler=SubsetRandomSampler(test_idx),
                                     num_workers=train_params['num_workers'])
    
        with torch.no_grad():
            test_losses = []

            for sample_batch in test_dataloader:
                cloud_batch, label_batch = sample_batch
                label_batch = label_batch.view(label_batch.shape[0], -1).float()
                cloud_batch, label_batch = cloud_batch.to(device), label_batch.to(device)

                output = pointnet_model(cloud_batch)
                loss = criterion(output, label_batch)
                test_losses.append(loss.item())
        
            print('Test loss:', np.mean(test_losses), '+/-', np.std(test_losses))
            print('Test loss (scaled):', np.mean(test_losses) * tgt_norm_val, 
                '+/-', np.std(test_losses) * tgt_norm_val)

        folds_losses.append(np.mean(test_losses))
        fold_count += 1
    
    print('\n>> FOLDS LOSS:', np.mean(folds_losses), '+/-', np.std(folds_losses))
    print('>> FOLDS LOSS (SCALED):', np.mean(folds_losses) * tgt_norm_val, 
        '+/-', np.std(folds_losses) * tgt_norm_val)

    return pointnet_model, folds_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PointNet for regression.')
    parser.add_argument('finger', type=str, help="Finger's sensor [MF, TH]")
    parser.add_argument('target', type=str, help="Tactile target [PDC, ELEC]")
    parser.add_argument('--cv', action='store_true', help="Perform 5-CV")
    args = parser.parse_args()

    tgt_cols, tgt_norm_val = get_task_params(args.finger, args.target)
    print('>> FINGER', args.finger, '- COLS', tgt_cols, '- NORM', tgt_norm_val)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('>> USING DEVICE', device)

    # Training parameters
    model_params = {
        'downsample_size' : 500,
        'batch_size' : 5,
        'num_workers' : 4,
        'cloud_fields' : 3,
        'learning_rate' : 0.01,
        'epochs' : 50,
        'device' : device
    }

    if args.cv:
        model, folds_losses = train_model_cv(tgt_cols, tgt_norm_val, model_params)
    else:
        model, train_losses = train_model(tgt_cols, tgt_norm_val, model_params)
        test_losses = test_model(tgt_cols, tgt_norm_val, model_params, model)