# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


# class Custom(data.Dataset):
#     def __init__(self, data_path, attr_path, image_size):
#         self.data_path = data_path
#         att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
#         atts = [att_list.index(att) + 1 for att in selected_attrs]
#         self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
#         self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

#         self.tf = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#     def __getitem__(self, index):
#         img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])).convert('RGB'))
#         att = torch.tensor((self.labels[index] + 1) // 2)
#         return img, att

#     def __len__(self):
#         return len(self.images)

def make_dataset(root, mode):
    assert mode in ['train', 'valid', 'test']
    lines = [line.rstrip() for line in open(os.path.join(root, 'attributes.txt'), 'r')]
    all_attr_names = lines[0].strip().split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[1:]
    train_font_num = 120
    val_font_num = 28  # noqa
    char_num = 52
    train_size = train_font_num * char_num
    if mode == 'train':
        lines = lines[:train_size]
    else:
        lines = lines[train_size:]

    items = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0].strip()
        values = split[1:]
        label = []
        for val in values:
            label.append((float(val)/100) >= 0.5)
        items.append([filename, label])
    return items


class Explo(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode):
        self.root = data_path
        self.items = make_dataset(self.root, mode)
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, 'image', filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)

# class CelebA(data.Dataset):
#     def __init__(self, data_path, attr_path, image_size, mode):
#         super(CelebA, self).__init__()
#         self.data_path = data_path
#         att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
#         atts = [att_list.index(att) + 1 for att in selected_attrs]
#         images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
#         labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

#         if mode == 'train':
#             self.images = images[:182000]
#             self.labels = labels[:182000]
#         if mode == 'valid':
#             self.images = images[182000:182637]
#             self.labels = labels[182000:182637]
#         if mode == 'test':
#             self.images = images[182637:]
#             self.labels = labels[182637:]

#         self.tf = transforms.Compose([
#             transforms.CenterCrop(170),
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

#         self.length = len(self.images)

#     def __getitem__(self, index):
#         img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
#         att = torch.tensor((self.labels[index] + 1) // 2)
#         return img, att

#     def __len__(self):
#         return self.length


# def check_attribute_conflict(att_batch, att_name, att_names):
#     def _get(att, att_name):
#         if att_name in att_names:
#             return att[att_names.index(att_name)]
#         return None

#     def _set(att, value, att_name):
#         if att_name in att_names:
#             att[att_names.index(att_name)] = value
#     att_id = att_names.index(att_name)
#     for att in att_batch:
#         if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
#             if _get(att, 'Bangs') != 0:
#                 _set(att, 1-att[att_id], 'Bangs')
#         elif att_name == 'Bangs' and att[att_id] != 0:
#             for n in ['Bald', 'Receding_Hairline']:
#                 if _get(att, n) != 0:
#                     _set(att, 1-att[att_id], n)
#                     _set(att, 1-att[att_id], n)
#         elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
#             for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
#                 if n != att_name and _get(att, n) != 0:
#                     _set(att, 1-att[att_id], n)
#         elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
#             for n in ['Straight_Hair', 'Wavy_Hair']:
#                 if n != att_name and _get(att, n) != 0:
#                     _set(att, 1-att[att_id], n)
#         elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
#             for n in ['Mustache', 'No_Beard']:
#                 if n != att_name and _get(att, n) != 0:
#                     _set(att, 1-att[att_id], n)
#     return att_batch


if __name__ == '__main__':
    import argparse
    # import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    # attrs_default = [
    #     'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    #     'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    # ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=None, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()

    dataset = Explo(args.data_path, args.attr_path, 64, 'valid')
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y

    dataset = Explo(args.data_path, args.attr_path, 64, 'valid')
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )
