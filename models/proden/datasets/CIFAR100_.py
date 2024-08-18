from PIL import Image
import os
import os.path
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
import random
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url

class IMBALANCECIFAR100(data.Dataset):
    """CIFAR-100 Dataset."""

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    cls_num = 100

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
                    self.train_labels += entry['fine_labels']

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
                self.test_labels = entry['fine_labels']

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

        with tarfile.open(os.path.join(self.root, self.filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

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