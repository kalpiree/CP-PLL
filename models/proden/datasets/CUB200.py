from PIL import Image
import os
import os.path
import torch
import torch.utils.data as data
import numpy as np
from torchvision import transforms
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url

class CUB200(data.Dataset):
    """CUB-200-2011 Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``CUB_200_2011`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'CUB_200_2011'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    # base_folder = 'CUB_200_2011'
    # url = 'https://data.caltech.edu/tindfiles/serve/d7aede8a-0926-4406-b7c0-3c76026c1f83/'
    # filename = 'CUB_200_2011.tgz'
    # tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.dataset = 'cub200'

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.load_dataset()

        if self.train:
            if partial_rate != 0.0:
                y = binarize_class(torch.tensor(self.image_labels))
                self.image_final_labels, self.average_class_label = partialize(y, torch.tensor(self.image_labels), partial_type, partial_rate)
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

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder, 'images.txt'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.base_folder, 'images', self.image_paths[index])
        img = Image.open(img_path).convert('RGB')
        target = self.image_final_labels[index]

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #
        # return img, target, index
        # target = self.image_final_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # For compatibility with your main script
        true_label = target  # or set it to None if you don't have a true value

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
