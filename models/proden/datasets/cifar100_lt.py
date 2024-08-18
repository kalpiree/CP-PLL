from PIL import Image
import os
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
from utils.utils_algo import download_url, check_integrity, binarize_class, partialize


class cifar100(data.Dataset):
    base_folder = 'cifar-100-python'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    val_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train_or_not=True, val_or_not=False, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.01, imbalance=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.validation = val_or_not
        self.dataset = 'cifar100'
        self.imb_factor = imb_factor
        self.imbalance = imbalance

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.data = []
            self.labels = []
            file = os.path.join(self.root, self.base_folder, self.train_list[0][0])
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['fine_labels']

            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))

            num_train = int(0.8 * len(self.data))
            self.train_data = self.data[:num_train]
            self.train_labels = self.labels[:num_train]

            if imbalance:
                self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 100, 'exp', self.imb_factor)

            self.train_data = torch.from_numpy(self.train_data).byte()
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)

            if partial_rate != 0.0:
                y = binarize_class(self.train_labels)
                self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
            else:
                self.train_final_labels = binarize_class(self.train_labels).float()

        elif self.validation:
            self.val_data = []
            self.val_labels = []
            file = os.path.join(self.root, self.base_folder, self.val_list[0][0])
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.val_data.append(entry['data'])
                self.val_labels += entry['fine_labels']

            self.val_data = np.concatenate(self.val_data)
            self.val_data = self.val_data.reshape((50000, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

            num_train = int(0.8 * len(self.val_data))
            self.val_data = self.val_data[num_train:]
            self.val_labels = self.val_labels[num_train:]

            self.val_data = torch.from_numpy(self.val_data).byte()
            self.val_labels = torch.tensor(self.val_labels, dtype=torch.long)

        else:
            file = os.path.join(self.root, self.base_folder, self.test_list[0][0])
            with open(file, 'rb') as fo:
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            self.test_labels = entry['fine_labels']
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

            self.test_data = torch.from_numpy(self.test_data).byte()
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)

    def gen_imbalanced_data(self, data, targets, num_classes, imb_type='exp', imb_factor=0.01):
        img_max = len(data) / num_classes
        img_num_per_cls = self.get_img_num_per_cls(num_classes, imb_type, imb_factor, img_max)

        new_data = []
        new_targets = []
        targets_np = np.array(targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)

        new_data = np.vstack(new_data)
        new_targets = np.array(new_targets)

        return new_data, new_targets

    def get_img_num_per_cls(self, num_classes, imb_type, imb_factor, img_max):
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(num_classes):
                num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls = [int(img_max)] * num_classes
        return img_num_per_cls

    def __getitem__(self, index):
        if self.train:
            img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
        elif self.validation:
            img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
        else:
            img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, true, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.validation:
            return len(self.val_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.val_list + self.test_list):
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
        tmp = 'train' if self.train else 'val' if self.validation else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(tmp, self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
