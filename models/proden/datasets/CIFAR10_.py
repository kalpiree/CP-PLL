from PIL import Image
import os
import os.path
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
import random
import torchvision.transforms as transforms
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url

class CIFAR10(data.Dataset):
    cls_num = 10

    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0, imb_type=None, imb_factor=1.0, reverse=False, shuffle=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.partial_type = partial_type
        self.partial_rate = partial_rate
        self.imb_type = imb_type
        self.imb_factor = imb_factor

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_labels += entry['labels']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            self.train_data = torch.from_numpy(self.train_data)
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)

            if self.imb_type is not None:
                np.random.seed(random_state)
                img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, reverse, shuffle)
                self.gen_imbalanced_data(img_num_list)

            if self.partial_rate != 0.0:
                y = binarize_class(self.train_labels)
                self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, self.partial_type, self.partial_rate)
            else:
                self.train_final_labels = binarize_class(self.train_labels).float()
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            self.test_labels = entry['labels']
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

            self.test_data = torch.from_numpy(self.test_data)
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse, shuffle=False):
        img_max = len(self.train_data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        if shuffle:
            random.shuffle(img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.train_labels, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.train_data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.train_data = new_data
        self.train_labels = torch.tensor(new_targets, dtype=torch.long)

    # def __getitem__(self, index):
    #     if self.train:
    #         img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
    #     else:
    #         img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
    #
    #     img = Image.fromarray(img.numpy(), mode=None)
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return img, target, true, index

    def __getitem__(self, index):
        if self.train:
            img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
        else:
            img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]

        img = Image.fromarray(img.numpy() if isinstance(img, torch.Tensor) else img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, true, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    base_folder = 'cifar-10-batches-py'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='CIFAR-10 Dataset Loader')
#     parser.add_argument('--imbalance', action='store_true', help='Use imbalanced dataset')
#     parser.add_argument('--imb_type', type=str, default='exp', choices=['exp', 'step'], help='Imbalance type')
#     parser.add_argument('--imb_factor', type=float, default=0.01, help='Imbalance factor')
#     parser.add_argument('--reverse', action='store_true', help='Reverse imbalance order')
#     parser.add_argument('--shuffle', action='store_true', help='Shuffle the number of images per class')
#     args = parser.parse_args()
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     if args.imbalance:
#         trainset = CIFAR10(root='./data', train_or_not=True, download=True, transform=transform,
#                            imb_type=args.imb_type, imb_factor=args.imb_factor, reverse=args.reverse, shuffle=args.shuffle)
#     else:
#         trainset = CIFAR10(root='./data', train_or_not=True, download=True, transform=transform)
#
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
#
#     for batch_idx, (data, target, true, index) in enumerate(trainloader):
#         print(f'Batch {batch_idx}:')
#         print(f'  Data shape: {data.shape}')
#         print(f'  Target shape: {target.shape}')
#         print(f'  True labels shape: {true.shape}')
#         break
