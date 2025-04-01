import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread
import pandas as pd
from datetime import datetime
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
from dataset_classes import Dataset_train
import archs_classes as archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import cv2

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

Fold = 1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='HarDMSEG',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: HarDMSEG)')
    parser.add_argument('--deepsupervision', default='False', type=str2bool)
    parser.add_argument('--dataset', default='keypoint',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='dcm',
                        help='mask file extension')
    parser.add_argument('--aug', default='True', type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default= 600, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=650, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'Ranger'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD', 'Ranger']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--m_class', default=4, type=float,
                        help='4')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, criterion_class, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    loss_h = AverageMeter()
    loss_c = AverageMeter()

    ious = AverageMeter()
    dist_loss = AverageMeter()
    model.train()

    for i, (input, target, class_label) in enumerate(train_loader): #, total=len(train_loader):
        input = input.cuda()
        target = target.cuda()
        class_label.cuda()
        optimizer.zero_grad()
        # compute output
        if args.deepsupervision:

        else:
            output, class_out = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion_class(class_out.cuda(), class_label.long().cuda())
            loss = loss1 + loss2
            iou = iou_score(output, target)
            
        loss_h.update(loss1.item(), input.size(0))
        loss_c.update(loss2.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dist_loss.update(0, input.size(0))
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss_heatmap', loss_h.avg),
        ('loss_class', loss_c.avg),
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dist_loss',dist_loss.avg)
    ])

    return log

def validate(args, val_loader, model, criterion, criterion_class):
    losses = AverageMeter()
    loss_h = AverageMeter()
    loss_c = AverageMeter()
    ious = AverageMeter()
    dist_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target, class_label) in enumerate(val_loader): #, total=len(train_loader):
            input = input.cuda()
            target = target.cuda()
            class_label = class_label.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                dst = 0
                tmloss = 0
                tmdst = 0
                for output in outputs:
                    tmloss, tmdst = criterion(output, target)
                    loss += tmloss
                    dst += tmdst
                loss /= len(outputs)
                dst  /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output, class_out = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion_class(class_out.cuda(), class_label.long().cuda())
                loss  = loss1 + loss2
                iou = iou_score(output, target)
            
            loss_h.update(loss1.item(), input.size(0))
            loss_c.update(loss2.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dist_loss.update(0, input.size(0))

    log = OrderedDict([
        ('loss_heatmap', loss_h.avg),
        ('loss_class', loss_c.avg),
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dist_loss',dist_loss.avg)
    ])

    return log


def main():
    F5folder = [[1]]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for folder_time in F5folder:
        args = parse_args()
        criterion = nn.MSELoss(reduce = True, size_average = False)
        criterion_class = nn.CrossEntropyLoss()

        cudnn.benchmark = True

        scale_tensor_transform = transforms.Compose([            
                transforms.ToTensor(),
                lambda x: x.float(),
            ])
    
        if args.name is None:
            if args.deepsupervision:
                args.name = '%s_%s_%s_wDS' %(args.dataset, args.arch,str(folder_time[4]))
            else:
                args.name = '%s_%s_%s_woDS' %(args.dataset, args.arch,str(folder_time[4]))

        if not os.path.exists('Fold'+str(Fold)+'_models/%s' %args.name):
            os.makedirs('Fold'+str(Fold)+'_models/%s' %args.name)

        print('Config -----')
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)))

        print('------------')
    	
        with open('Fold'+str(Fold)+'_models/%s/args.txt' %args.name, 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f)

        joblib.dump(args, 'Fold'+str(Fold)+'_models/%s/args.pkl' %args.name)
      
        train_img_paths = []
        train_mask_paths = []
        
        train_img_paths = glob('C:\\Users\\user\\brain_hardnet_only\\Data\\brain_classified\\img_train80\\*')
        train_mask_paths = glob('C:\\Users\\user\\brain_hardnet_only\\Data\\brain_classified\\lab_val80\\*')

        val_img_paths = glob('C:\\Users\\user\\brain_hardnet_only\\Data\\brain_classified\\img_train20\\*')
        val_mask_paths = glob('C:\\Users\\user\\brain_hardnet_only\\Data\\brain_classified\\lab_val20\\*') 

        print(len(train_img_paths))
        print(len(val_img_paths))
        # create model
    
        print("=> creating model %s" %args.arch)
        model = archs.__dict__[args.arch](args)
        
        model = nn.DataParallel(model)
        model.to(device)

        torch.save(model.state_dict(), 'fack_model.pth')

        print(count_params(model))
        
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGDP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Ranger':
            optimizer = optim.RangerVA(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=args.weight_decay)

        train_dataset = Dataset_train(args, train_img_paths, train_mask_paths, scale_tensor_transform, args.aug)
        val_dataset = Dataset_train(args, val_img_paths, val_mask_paths, scale_tensor_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'train_loss', 'train_iou', 'train_dist', 'val_loss', 'val_iou', 'val_dist'
        ])

        best_ds = 9999
        trigger = 0
        for epoch in range(args.epochs):
            print('val folder :%s - Epoch [%d/%d]' %(str(folder_time[0]),epoch, args.epochs))
            train_log = train(args, train_loader, model, criterion, criterion_class, optimizer, epoch)
            
            # evaluate on validation set
            val_log = validate(args, val_loader, model, criterion, criterion_class)
            
            print('loss %.4f - iou %.4f - dist %.4f - val_loss %.4f - val_iou %.4f - val_dist %.4f ' 
                %(train_log['loss'], train_log['iou'], train_log['dist_loss'], val_log['loss'], val_log['iou'], val_log['dist_loss']))

            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                train_log['loss_class'],
                train_log['loss_heatmap'],
                train_log['iou'],
                train_log['dist_loss'],
                val_log['loss'],
                val_log['loss_class'],
                val_log['loss_heatmap'],                
                val_log['iou'],
                val_log['dist_loss']
            ], index=['epoch', 'lr', 'train_loss', 'train_loss_class', 'train_loss_heatmap','train_iou','train_dist',
                                     'val_loss', 'val_loss_class', 'val_loss_heatmap','val_iou', 'val_dist'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('Fold'+str(Fold)+'_models/%s/brain.csv' %args.name, index=False)

            trigger += 1

            torch.save(model.state_dict(), 'Fold'+str(Fold)+'_models/%s/train_model_%s.pth' %(args.name,str(epoch)))

            if val_log['loss'] < best_ds:
                torch.save(model.state_dict(), 'Fold'+str(Fold)+'_models/%s/model_%s.pth' %(args.name,str(epoch)))
                torch.save(model.state_dict(), 'Fold'+str(Fold)+'_models/%s/model_best.pth' %args.name)
                best_ds = val_log['loss']
                print("=> saved best model")
                trigger = 0
            
            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()
    

if __name__ == '__main__':
    main()
