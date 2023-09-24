import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random


class SirstDataset(Data.Dataset):

    def __init__(self, args, mode='train'):

        base_dir = 'datasets/SIRST'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.35619214, 0.35575104, 0.35673013], [0.2614548, 0.26135704, 0.26168558]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_pixels0.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_size = self.img_size
        # random scale (short edge)
        long_size = random.randint(int(self.img_size * 0.5), int(self.img_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < img_size:
            padh = img_size - oh if oh < img_size else 0
            padw = img_size - ow if ow < img_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - img_size)
        y1 = random.randint(0, h - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        mask = mask.crop((x1, y1, x1 + img_size, y1 + img_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _val_sync_transform(self, img, mask):

        outsize = self.img_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))

        return img, mask

    def _testval_sync_transform(self, img, mask):
        img_size = self.img_size
        img = img.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)

        return img, mask


class IRSTD1K_Dataset(Data.Dataset):

    def __init__(self, args, mode='train'):

        base_dir = 'datasets/IRSTD-1k'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_size = self.img_size
        # random scale (short edge)
        long_size = random.randint(int(self.img_size * 0.5), int(self.img_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < img_size:
            padh = img_size - oh if oh < img_size else 0
            padw = img_size - ow if ow < img_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - img_size)
        y1 = random.randint(0, h - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        mask = mask.crop((x1, y1, x1 + img_size, y1 + img_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _val_sync_transform(self, img, mask):

        outsize = self.img_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))

        return img, mask

    def _testval_sync_transform(self, img, mask):
        img_size = self.img_size
        img = img.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)

        return img, mask


def get_mean_std(data_set):
    loader = Data.DataLoader(data_set, batch_size=16, num_workers=1, shuffle=False, pin_memory=True)

    sum_of_pixels = torch.zeros(3)
    sum_of_square_error = torch.zeros(3)
    pixels_per_channel = len(data_set) * 480 * 480
    for X, _ in loader:
        for d in range(3):
            sum_of_pixels[d] += X[:, d, :, :].sum()
    _mean = sum_of_pixels / pixels_per_channel
    for X, _ in loader:
        for d in range(3):
            sum_of_square_error[d] += ((X[:, d, :, :] - _mean[d]).pow(2)).sum()
    _std = torch.sqrt(sum_of_square_error / pixels_per_channel)

    return list(_mean.numpy()), list(_std.numpy())
