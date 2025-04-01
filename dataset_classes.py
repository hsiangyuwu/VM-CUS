import numpy as np
import cv2
import random
import pydicom as pyd
from skimage.io import imread
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import os
from PIL import Image
import copy
import scipy.io

class Dataset_train(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        seed = np.random.randint(48651351)
        #random.seed(seed)
        #torch.manual_seed(seed)
        transform_train = transforms.Compose([
                                 transforms.ToPILImage(),   
                                 transforms.RandomAffine(degrees=[-20,20], translate=[0.1, 0.1], scale=[0.90, 1.1], shear=[-10,10]),
                                 transforms.ToTensor(),
                                ])

        imageMat = scipy.io.loadmat(img_path)
        image = imageMat['Img']
        maskMat = scipy.io.loadmat(mask_path)
        mask = maskMat['m_class']

        image = image.astype('float32')
        for idx in range(0, 3):
            image2 = image[:, :, idx]
            image2 = np.array(image2)
            random.seed(seed)
            torch.manual_seed(seed)
            image2 = transform_train(image2)
            if idx == 0:
                image3 = image2
            else:
                image2 = image2
                image3 = torch.cat((image3, image2),dim=0)
        image = np.asarray(image3)


        mask = mask.astype('float32')
        for idx in range(0, 4):
            mask2 = mask[idx, :, :]
            mask2 = np.array(mask2)
            random.seed(seed)
            torch.manual_seed(seed)
            mask2 = transform_train(mask2)
            if idx == 0:
                mask3 = mask2
            else:
                mask2 = mask2
                mask3 = torch.cat((mask3, mask2),dim=0)
        mask = np.asarray(mask3)

        ext1 = img_path.rindex('\\')
        label = img_path[ext1+1:ext1+3]
        class_label = torch.from_numpy(np.array(float(label.replace('_', ''))))

        return image, mask, class_label

class Dataset_test(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, transform, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.aug = aug
        self.transform = transform        
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        imageMat = scipy.io.loadmat(img_path)
        image = imageMat['Img']
        image = image.astype('float32')
        image = image.transpose((2, 0, 1))        
        image = np.asarray(image)
        
        
        ext1 = img_path.rindex('\\')
        label = img_path[ext1+1:ext1+3]
        class_label = torch.from_numpy(np.array(float(label.replace('_', ''))))

        return image, class_label




