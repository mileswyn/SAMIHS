import os
from random import randint
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
import random
from torch.utils.data import DataLoader
from scipy.ndimage import label as clabel
from tqdm import tqdm


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask != 0)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask == 0)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask != 0)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask == 0)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2] 
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    shiftw = randint(-int(0.9*img_size), int(1.1*img_size))
    shifth = randint(-int(0.9*img_size), int(1.1*img_size))
    shiftx = randint(-int(0.05*img_size), int(0.05*img_size))
    shifty = randint(-int(0.05*img_size), int(0.05*img_size))

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask != 0)
    # indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])
    
class Transform2D_BCIHM:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, mode='train', img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.5, p_rota=0.5, p_scale=0.0, p_gaussn=1.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.mode = mode
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):

        # transforming to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)

        # if self.mode == 'train':
        # # random horizontal flip
        #     if np.random.rand() < self.p_flip:
        #         image, mask = F.hflip(image), F.hflip(mask)

        #     # random rotation
        #     if np.random.rand() < self.p_rota:
        #         angle = T.RandomRotation.get_params((-30, 30))
        #         image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        #     # random add gaussian noise
        #     if np.random.rand() < self.p_gaussn:
        #         image, mask = image.cpu().numpy().transpose(1,2,0), image.cpu().numpy().transpose(1,2,0)
        #         ns = np.random.randint(3, 15)
        #         noise = np.random.normal(loc=0, scale=1, size=(512, 512, 1)) * ns
        #         noise = noise.astype(int)
        #         image = np.array(image) + noise
        #         image, mask = F.to_tensor(image), F.to_tensor(mask)

        # else:
        #     pass

        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = (image - image.min()) / (image.max() - image.min())

        return image, mask, low_mask
    
class BCIHM(Dataset):
    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, fold=0, img_size=256, prompt = "click", class_id=1,
                 one_hot_mask: int = False) -> None:
        self.fold = fold
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        id_list_file = os.path.join('./dataset/excel', 'BCIHM.csv')
        df = pd.read_csv(id_list_file, encoding='gbk')
        # id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        if self.split == 'train':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] != self.fold and df['label'][id] > 0] 
            self.gt_list = [label for id, label in enumerate(df['gt']) if df['fold'][id] != self.fold and df['label'][id] > 0]
        elif self.split == 'val':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        elif self.split == 'test':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.classes = 2
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, i):
        """Get the images"""
        name = self.img_list[i]
        img_path = os.path.join(self.dataset_path, name)
        
        mask_name = self.gt_list[i]
        msk_path = os.path.join(self.dataset_path, mask_name)

        image = np.load(img_path)
        mask = np.load(msk_path)

        class_id = 1  # fixed since only one class of foreground 
        mask[mask > 0] = 1

        image = np.clip(image, np.percentile(image, 0.05), np.percentile(image, 99.5)).astype(np.int16)
        mask = mask.astype(np.uint8)
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
            mask, low_mask = mask.squeeze(0), low_mask.squeeze(0)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

         # --------- make the point prompt ----------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            pt = pt * self.img_size / 512
            mask[mask!=0] = 1
            mask[mask!=1] = 0
            low_mask[low_mask!=0] = 1
            low_mask[low_mask!=1] = 0
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bbox = bbox * self.img_size / 512
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': name.split('/')[-1].split('.')[0] + '.png',
            'class_id': class_id,
            }
    
class Transform2D_Instance:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):

        # transforming to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)

        # if self.mode == 'train':
        # # random horizontal flip
        #     if np.random.rand() < self.p_flip:
        #         image, mask = F.hflip(image), F.hflip(mask)

        #     # random rotation
        #     if np.random.rand() < self.p_rota:
        #         angle = T.RandomRotation.get_params((-30, 30))
        #         image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        #     # random add gaussian noise
        #     if np.random.rand() < self.p_gaussn:
        #         image, mask = image.cpu().numpy().transpose(1,2,0), image.cpu().numpy().transpose(1,2,0)
        #         ns = np.random.randint(3, 15)
        #         noise = np.random.normal(loc=0, scale=1, size=(512, 512, 1)) * ns
        #         noise = noise.astype(int)
        #         image = np.array(image) + noise
        #         image, mask = F.to_tensor(image), F.to_tensor(mask)

        # else:
        #     pass

        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = (image - image.min()) / (image.max() - image.min())

        return image, mask, low_mask

class Instance(Dataset):
    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, fold=0, img_size=256, prompt = "click", class_id=1,
                 one_hot_mask: int = False) -> None:
        self.fold = fold
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        id_list_file = os.path.join('./dataset/excel', 'Instance.csv')
        df = pd.read_csv(id_list_file, encoding='gbk')
        # id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        if self.split == 'train':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] != self.fold and df['label'][id] > 0] 
            self.gt_list = [label for id, label in enumerate(df['gt']) if df['fold'][id] != self.fold and df['label'][id] > 0]
        elif self.split == 'val':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        elif self.split == 'test':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.classes = 2
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, i):
        """Get the images"""
        name = self.img_list[i]
        img_path = os.path.join(self.dataset_path, name)
        
        mask_name = self.gt_list[i]
        msk_path = os.path.join(self.dataset_path, mask_name)

        image = np.load(img_path)
        mask = np.load(msk_path)

        class_id = 1  # fixed since only one class of foreground 
        mask[mask > 0] = 1

        image = np.clip(image, np.percentile(image, 0.05), np.percentile(image, 99.5)).astype(np.int16)
        mask = mask.astype(np.uint8)
        image, mask = correct_dims(image, mask)  
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
            mask, low_mask = mask.squeeze(0), low_mask.squeeze(0)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            pt = pt * self.img_size / 512
            mask[mask!=0] = 1
            mask[mask!=1] = 0
            low_mask[low_mask!=0] = 1
            low_mask[low_mask!=1] = 0
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bbox = bbox * self.img_size / 512
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': name.split('/')[-1].split('.')[0] + '.png',
            'class_id': class_id,
            }

class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)

if __name__ == '__main__':
    tf_val = Transform2D_BCIHM(mode='train', img_size=1024, low_img_size=256, ori_size=512, crop=None, p_flip=1, color_jitter_params=None, long_mask=True)
    val_dataset = BCIHM("/data/openData_Med/BCIHM/", "test", tf_val, img_size=1024, class_id=1)
    Idataset = Instance("/data/openData_Med/Instance/", "test", tf_val, img_size=1024, class_id=1)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    with tqdm(total=len(valloader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_idx, (datapack) in enumerate(valloader):
            pass
