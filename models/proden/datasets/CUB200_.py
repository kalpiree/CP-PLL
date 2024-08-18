from PIL import Image
import os
import torch
import torch.utils.data as data
import numpy as np
import random
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url


class IMBALANCECUB200(data.Dataset):
    """CUB-200-2011 Dataset with Imbalance Handling."""

    base_folder = 'CUB_200_2011'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    cls_num = 200

    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0, imb_type=None, imb_factor=1.0,
                 reverse=False, shuffle=False):
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

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.load_dataset()

        if self.train:
            if self.imb_type is not None:
                np.random.seed(random_state)
                img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, reverse, shuffle)
                self.gen_imbalanced_data(img_num_list)

            if partial_rate != 0.0:
                y = binarize_class(torch.tensor(self.image_labels))
                self.image_final_labels, self.average_class_label = partialize(y, torch.tensor(self.image_labels),
                                                                               partial_type, partial_rate)
            else:
                self.image_final_labels = binarize_class(torch.tensor(self.image_labels)).float()
        else:
            self.image_final_labels = self.image_labels

    def load_dataset(self):
        images_file = os.path.join(self.root, self.base_folder, 'images.txt')
        labels_file = os.path.join(self.root, self.base_folder, 'image_class_labels.txt')
        train_test_split_file = os.path.join(self.root, self.base_folder, 'train_test_split.txt')

        with open(images_file, 'r') as f:
            image_paths = [line.strip().split(' ')[1] for line in f.readlines()]

        with open(labels_file, 'r') as f:
            image_labels = [int(line.strip().split(' ')[1]) - 1 for line in f.readlines()]

        with open(train_test_split_file, 'r') as f:
            is_training_image = [line.strip().split(' ')[1] == '1' for line in f.readlines()]

        if self.train:
            self.image_paths = [image_paths[i] for i in range(len(image_paths)) if is_training_image[i]]
            self.image_labels = [image_labels[i] for i in range(len(image_labels)) if is_training_image[i]]
        else:
            self.image_paths = [image_paths[i] for i in range(len(image_paths)) if not is_training_image[i]]
            self.image_labels = [image_labels[i] for i in range(len(image_labels)) if not is_training_image[i]]

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse, shuffle=False):
        img_max = len(self.image_paths) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
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
        new_image_paths = []
        new_image_labels = []
        targets_np = np.array(self.image_labels, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_image_paths.extend([self.image_paths[i] for i in selec_idx])
            new_image_labels.extend([the_class] * the_img_num)
        self.image_paths = new_image_paths
        self.image_labels = new_image_labels

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder, 'images.txt'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.base_folder, 'images', self.image_paths[index])
        img = Image.open(img_path).convert('RGB')
        target = self.image_final_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        true_label = self.image_labels[index]

        return img, target, true_label, index

    def __len__(self):
        return len(self.image_paths)

    def download(self):
        import tarfile

        if self._check_exists():
            print('Files already downloaded and verified')
            return

        # If we have already manually downloaded the file, just extract it
        fpath = os.path.join(self.root, self.filename)
        if os.path.isfile(fpath):
            print(f'Extracting {fpath}')
            with tarfile.open(fpath, 'r:gz') as tar:
                tar.extractall(path=self.root)
            return

        print(f'Downloading {self.url} to {fpath}')
        download_url(self.url, self.root, self.filename, self.tgz_md5)

        print(f'Extracting {fpath}')
        with tarfile.open(fpath, 'r:gz') as tar:
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
