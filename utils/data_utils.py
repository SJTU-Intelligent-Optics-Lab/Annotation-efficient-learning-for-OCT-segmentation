
import math
import os
import cv2
import numpy as np
import torch
from monai import data, transforms
import PIL
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    if args.test_mode:
        dataset_test = build_dataset(args.val_text, args=args)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_test, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False)
        loader = data_loader_val

    else:
        dataset_train = build_dataset(args.train_text, args=args)
        dataset_val = build_dataset(args.val_text, args=args)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle = True,
            drop_last=False)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False)
        loader = [data_loader_train, data_loader_val]

    return loader

def build_dataset(split, args):
    transform = build_transform(split, args)

    if 'fewshot' in split:
        dataset = Fewshot_dataset(args.label_index, args.data_dir, args.list_dir, split, transform=transform)
    else:
        dataset = Parent_dataset(args.data_dir, args.list_dir, split, transform=transform)
    print(dataset)
    return dataset

def build_transform(split, args):
    if 'train' in split:
        transform_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.input_size,args.input_size), interpolation=PIL.Image.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(p=0.5),#
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.227, 0.227, 0.227], std=[0.1935, 0.1935, 0.1935])])

        transform_label = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomRotation(degrees=(-180, 180)), #
            transforms.Resize((args.input_size,args.input_size), interpolation=PIL.Image.NEAREST),  # 3 is bicubic
            transforms.RandomHorizontalFlip(p=0.5),#
            transforms.ToTensor()])
        trans = [transform_image, transform_label]
        return trans

    elif 'val' in split:
        transform_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.input_size,args.input_size), interpolation=PIL.Image.BICUBIC),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.227, 0.227, 0.227], std=[0.1935, 0.1935, 0.1935])])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.input_size,args.input_size), interpolation=PIL.Image.NEAREST),  # 3 is bicubic
            transforms.ToTensor()])
        trans = [transform_image, transform_label]
        return trans

class Parent_dataset(Dataset):
    def __init__(self, data_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = data_dir
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        image_path = os.path.join(self.data_dir, self.split, slice_name)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        h, w =image.shape[0], image.shape[1]
        image = np.expand_dims(np.array(image), axis = -1).repeat(3,2)
        label_path = os.path.join(self.data_dir, self.split, slice_name.replace('.png','_gt.png'))

        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(np.array(label), axis = -1)

        if self.transform:
            image = self.transform[0](image)
            label = self.transform[1](label)
        if self.split == 'val':
            return [image, label, image_path, h, w]

        return [image, label]

class Fewshot_dataset(Dataset):
    def __init__(self, label_index, data_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = data_dir
        self.label_index=label_index

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        if 'train' in self.split:
            image_path = os.path.join(self.data_dir, 'train_fewshot_data', slice_name)
            label_path = os.path.join(self.data_dir, 'train_fewshot_data', slice_name.replace('.png', '_gt.png'))
        elif 'val' in self.split:
            image_path = os.path.join(self.data_dir, 'val_fewshot_data', slice_name)
            label_path = os.path.join(self.data_dir, 'val_fewshot_data', slice_name.replace('.png', '_gt.png'))
        else:
            raise 'can not find the train or val dict'
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        h, w =image.shape[0], image.shape[1]
        image = np.expand_dims(np.array(image), axis = -1).repeat(3,2)

        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)

        # for i in self.label_index:
        #     if i<255:
        #         label[label==i*25] = 255
        # label[label!=255] = 0

        label = np.expand_dims(np.array(label), axis = -1)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            image = self.transform[0](image)
            torch.random.manual_seed(seed)
            label = self.transform[1](label)

        if 'val' in self.split:
            return [image, label, image_path, h, w]

        return [image, label]
