import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from dataset_classes import Dataset_test
import cv2
import pandas as pd
import archs_classes as archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import joblib
import scipy.io
import re

class_number = 12
keypoints_n = 4
Fold = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savename', default='Brain', # Brain
                        help='model name')
    parser.add_argument('--FolderName', default='Brain',
                        help='model name')
    parser.add_argument('--name', default='',
                        help='model name')
    parser.add_argument('--fold', default=1,
                        help='fold name')
    args = parser.parse_args()

    return args

def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals

def main():
    val_args = parse_args()
    val_args.name = val_args.FolderName

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    args = joblib.load('Fold'+str(Fold)+'_models/%s/args.pkl' %val_args.name)
    args.batch_size = 1
    
    if not os.path.exists('Fold'+str(Fold)+'_output/%s_Test_mat' %val_args.savename):
        os.makedirs('Fold'+str(Fold)+'_output/%s_Test_mat' %val_args.savename)

    if not os.path.exists('Fold'+str(Fold)+'_output/%s_Test_csv' %val_args.savename):
        os.makedirs('Fold'+str(Fold)+'_output/%s_Test_csv' %val_args.savename)
        
    if not os.path.exists('Fold'+str(Fold)+'_output/%s_GoundTruth_csv' %val_args.name):
        os.makedirs('Fold'+str(Fold)+'_output/%s_GoundTruth_csv' %val_args.name)        

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'Fold'+str(Fold)+'_models/%s/args.pkl' %val_args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)
    model = nn.DataParallel(model)
    model.to(device)

    val_img_paths = glob('C:\\Users\\user\\brain_hardnet_only\\Data\\202205_avis\\*')

    scale_tensor_transform = transforms.Compose([            
            transforms.ToTensor(),
            lambda x: x.float(),
        ])

    model.load_state_dict(torch.load('Fold'+str(Fold)+'_models/Brain/best_model/train_model_182.pth'))
    model.eval()

    val_dataset = Dataset_test(args, val_img_paths, scale_tensor_transform)  # mat file

    print('batch_size: ' + str(args.batch_size))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with torch.no_grad():
            for i, (input, img_path) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.cuda()
                output, class_out = model(input)
                output = output.cpu().detach().numpy()
                nnsoftmax = nn.Softmax(dim=1)
                class_out = nnsoftmax(class_out)
                class_out = class_out.cpu().detach().numpy()
                class_score = class_out[0]
                img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]

                for ij in range(0, 1):
                    temp_file_name = os.path.basename(img_paths[ij])                    
                    scipy.io.savemat('Fold'+str(Fold)+'_output/%s_Test_mat/'%val_args.savename+temp_file_name[0:-4] + '.mat', {'output':(output[ij,:,:,:]).astype('double')})
                    preds_tmp = output[ij,:,:,:]                    
                    preds_tmp  = preds_tmp[:,:,np.newaxis]
                    preds_tmp = preds_tmp.transpose((2, 0, 1, 3))
                    out_preds, out_maxvals = get_max_preds(preds_tmp)                    
                    out_preds = out_preds.squeeze()
                    out_maxvals = out_maxvals.squeeze()
                    log_keypoints = pd.DataFrame(columns=['Name', 'xpa', 'ypa', 'mvalue'])
                    log_classes = pd.DataFrame(columns=['class', 'score'])

                    for Pi in range(keypoints_n):
                        tmp1 = pd.Series([
                            'Point' + str(Pi + 1),
                            out_preds[Pi][0],
                            out_preds[Pi][1],
                            out_maxvals[Pi]
                        ], index=['Name', 'xpa', 'ypa', 'mvalue'])
                        log_keypoints = log_keypoints.append(tmp1, ignore_index=True)
                    
                    for Si in range(class_number):
                        tmp2 = pd.Series([
                            str(Si + 1),  # Class 
                            class_score[Si]  # Score 
                        ], index=['class', 'score'])
                        log_classes = log_classes.append(tmp2, ignore_index=True)

                    log = pd.concat([log_keypoints, log_classes], axis=1)
                    log.to_csv('Fold'+str(Fold)+'_output/%s_Test_csv/'%val_args.savename+temp_file_name[0:-4] + '.csv', index=False)

        torch.cuda.empty_cache()
   
if __name__ == '__main__':
    main()
