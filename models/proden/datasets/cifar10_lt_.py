# from PIL import Image
# import os
# import sys
# import torch
# import numpy as np
# import pickle
# import torch.utils.data as data
# from utils.utils_algo import binarize_class, partialize, check_integrity, download_url
# from sklearn.model_selection import train_test_split
#
# class cifar10(data.Dataset):
#     base_folder = 'cifar-10-batches-py'
#     url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#     filename = 'cifar-10-python.tar.gz'
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#
#     def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
#                  partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.01, validation_split=True):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train_or_not
#         self.dataset = 'cifar10'
#         self.imb_factor = imb_factor
#         self.validation_split = validation_split
#
#         if download:
#             self.download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         if self.train:
#             self.train_data = []
#             self.train_labels = []
#             for fentry in self.train_list:
#                 f = fentry[0]
#                 file = os.path.join(self.root, self.base_folder, f)
#                 with open(file, 'rb') as fo:
#                     entry = pickle.load(fo, encoding='latin1')
#                 self.train_data.append(entry['data'])
#                 self.train_labels += entry['labels']
#
#             self.train_data = np.concatenate(self.train_data)
#             self.train_data = self.train_data.reshape((50000, 3, 32, 32))
#             self.train_data = self.train_data.transpose((0, 2, 3, 1))
#
#             # if self.validation_split:
#             #     self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(self.train_data, self.train_labels, random_state)
#             #     self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp', self.imb_factor)
#             if self.validation_split:
#                 self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(
#                     self.train_data, self.train_labels, random_state)
#                 self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp',
#                                                                           self.imb_factor)
#                 self.val_data = self.val_data.transpose((0, 3, 1, 2))  # Transpose validation data to [3, 32, 32]
#             self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 10, 'exp', self.imb_factor)
#
#             self.train_data = torch.from_numpy(self.train_data).float()
#             self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
#
#             if partial_rate != 0.0:
#                 y = binarize_class(self.train_labels)
#                 self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
#             else:
#                 self.train_final_labels = binarize_class(self.train_labels).float()
#         else:
#             f = self.test_list[0][0]
#             file = os.path.join(self.root, self.base_folder, f)
#             with open(file, 'rb') as fo:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#             self.test_data = entry['data']
#             self.test_labels = entry['labels']
#             self.test_data = self.test_data.reshape((10000, 3, 32, 32))
#             self.test_data = self.test_data.transpose((0, 2, 3, 1))
#
#             self.test_data = torch.from_numpy(self.test_data).float()
#             self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
#
#     def create_validation_split(self, data, labels, random_state):
#         train_data, val_data, train_labels, val_labels = train_test_split(
#             data, labels, test_size=0.2, stratify=labels, random_state=random_state
#         )
#         return train_data, val_data, train_labels, val_labels
#
#     def gen_imbalanced_data(self, data, targets, num_classes, imb_type='exp', imb_factor=0.01):
#         img_max = len(data) / num_classes
#         img_num_per_cls = self.get_img_num_per_cls(num_classes, imb_type, imb_factor, img_max)
#
#         new_data = []
#         new_targets = []
#         targets_np = np.array(targets, dtype=np.int64)
#         classes = np.unique(targets_np)
#
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             new_data.append(data[selec_idx, ...])
#             new_targets.extend([the_class] * the_img_num)
#
#         new_data = np.vstack(new_data)
#         new_targets = np.array(new_targets)
#
#         return new_data, new_targets
#
#     def get_img_num_per_cls(self, num_classes, imb_type, imb_factor, img_max):
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(num_classes):
#                 num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls = [int(img_max)] * num_classes
#         return img_num_per_cls
#
#     def __getitem__(self, index):
#         if self.train:
#             if self.validation_split and index >= len(self.train_data):
#                 index -= len(self.train_data)
#                 img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
#             else:
#                 img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
#         else:
#             img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
#
#         img = Image.fromarray(img.numpy(), mode=None)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, true, index
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data) + (len(self.val_data) if self.validation_split else 0)
#         else:
#             return len(self.test_data)
#
#     def _check_integrity(self):
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True
#
#     def download(self):
#         import tarfile
#
#         if self._check_integrity():
#             return
#
#         download_url(self.url, self.root, self.filename, self.tgz_md5)
#
#         cwd = os.getcwd()
#         tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
#         os.chdir(self.root)
#         tar.extractall()
#         tar.close()
#         os.chdir(cwd)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         tmp = 'train' if self.train is True else 'test'
#         fmt_str += '    Split: {}\n'.format(tmp)
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str

from PIL import Image
import os
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url
from sklearn.model_selection import train_test_split

# class cifar10(data.Dataset):
#     base_folder = 'cifar-10-batches-py'
#     url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#     filename = 'cifar-10-python.tar.gz'
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#
#     def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
#                  partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.01, validation_split=True, imbalance=True):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train_or_not
#         self.dataset = 'cifar10'
#         self.imb_factor = imb_factor
#         self.validation_split = validation_split
#         self.imbalance = imbalance
#
#         if download:
#             self.download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         if self.train:
#             self.train_data = []
#             self.train_labels = []
#             for fentry in self.train_list:
#                 f = fentry[0]
#                 file = os.path.join(self.root, self.base_folder, f)
#                 with open(file, 'rb') as fo:
#                     entry = pickle.load(fo, encoding='latin1')
#                 self.train_data.append(entry['data'])
#                 self.train_labels += entry['labels']
#
#             self.train_data = np.concatenate(self.train_data)
#             self.train_data = self.train_data.reshape((50000, 3, 32, 32))
#             self.train_data = self.train_data.transpose((0, 2, 3, 1))
#
#             if self.validation_split:
#                 self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(
#                     self.train_data, self.train_labels, random_state)
#                 if self.imbalance:
#                     self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp', self.imb_factor)
#                 #self.val_data = self.val_data.transpose((0, 3, 1, 2))  # Transpose validation data to [3, 32, 32]
#                 self.val_data = self.val_data.transpose((0, 3, 1, 2)).astype(
#                     np.uint8)  # Transpose validation data to [3, 32, 32] and convert to uint8
#
#             if self.imbalance:
#                 self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 10, 'exp', self.imb_factor)
#
#             # self.train_data = torch.from_numpy(self.train_data).float()
#             # self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
#
#             self.train_data = torch.from_numpy(self.train_data).byte()
#             self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
#
#             if partial_rate != 0.0:
#                 y = binarize_class(self.train_labels)
#                 self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
#             else:
#                 self.train_final_labels = binarize_class(self.train_labels).float()
#         else:
#             f = self.test_list[0][0]
#             file = os.path.join(self.root, self.base_folder, f)
#             with open(file, 'rb') as fo:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#             self.test_data = entry['data']
#             self.test_labels = entry['labels']
#             self.test_data = self.test_data.reshape((10000, 3, 32, 32))
#             self.test_data = self.test_data.transpose((0, 2, 3, 1))
#
#             # self.test_data = torch.from_numpy(self.test_data).float()
#             # self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
#             self.test_data = torch.from_numpy(self.test_data).byte()
#             self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
#
#     def create_validation_split(self, data, labels, random_state):
#         train_data, val_data, train_labels, val_labels = train_test_split(
#             data, labels, test_size=0.2, stratify=labels, random_state=random_state
#         )
#         return train_data, val_data, train_labels, val_labels
#
#     def gen_imbalanced_data(self, data, targets, num_classes, imb_type='exp', imb_factor=0.01):
#         img_max = len(data) / num_classes
#         img_num_per_cls = self.get_img_num_per_cls(num_classes, imb_type, imb_factor, img_max)
#
#         new_data = []
#         new_targets = []
#         targets_np = np.array(targets, dtype=np.int64)
#         classes = np.unique(targets_np)
#
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             new_data.append(data[selec_idx, ...])
#             new_targets.extend([the_class] * the_img_num)
#
#         new_data = np.vstack(new_data)
#         new_targets = np.array(new_targets)
#
#         return new_data, new_targets
#
#     def get_img_num_per_cls(self, num_classes, imb_type, imb_factor, img_max):
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(num_classes):
#                 num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls = [int(img_max)] * num_classes
#         return img_num_per_cls
#
#     # def __getitem__(self, index):
#     #     if self.train:
#     #         if self.validation_split and index >= len(self.train_data):
#     #             index -= len(self.train_data)
#     #             img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
#     #         else:
#     #             img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
#     #     else:
#     #         img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
#     #
#     #     img = Image.fromarray(img.numpy(), mode=None)
#     #
#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #
#     #     if self.target_transform is not None:
#     #         target = self.target_transform(target)
#     #
#     #     return img, target, true, index
#
#     def __getitem__(self, index):
#         if self.train:
#             if self.validation_split and index >= len(self.train_data):
#                 index -= len(self.train_data)
#                 img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
#             else:
#                 img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
#         else:
#             img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
#
#         # Ensure the image is in uint8 format
#         img = img.numpy().astype(np.uint8)
#         img = Image.fromarray(img, mode=None)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, true, index
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data) + (len(self.val_data) if self.validation_split else 0)
#         else:
#             return len(self.test_data)
#
#     def _check_integrity(self):
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True
#
#     def download(self):
#         import tarfile
#
#         if self._check_integrity():
#             return
#
#         download_url(self.url, self.root, self.filename, self.tgz_md5)
#
#         cwd = os.getcwd()
#         tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
#         os.chdir(self.root)
#         tar.extractall()
#         tar.close()
#         os.chdir(cwd)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         tmp = 'train' if self.train is True else 'test'
#         fmt_str += '    Split: {}\n'.format(tmp)
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str

# from PIL import Image
# import os
# import sys
# import torch
# import numpy as np
# import pickle
# import torch.utils.data as data
# from utils.utils_algo import binarize_class, partialize, check_integrity, download_url
# from sklearn.model_selection import train_test_split
#
# class cifar10(data.Dataset):
#     base_folder = 'cifar-10-batches-py'
#     url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#     filename = 'cifar-10-python.tar.gz'
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#
#     def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
#                  partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.01, validation_split=True, imbalance=True):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train_or_not
#         self.dataset = 'cifar10'
#         self.imb_factor = imb_factor
#         self.validation_split = validation_split
#
#         if download:
#             self.download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         if self.train:
#             self.train_data = []
#             self.train_labels = []
#             for fentry in self.train_list:
#                 f = fentry[0]
#                 file = os.path.join(self.root, self.base_folder, f)
#                 with open(file, 'rb') as fo:
#                     entry = pickle.load(fo, encoding='latin1')
#                 self.train_data.append(entry['data'])
#                 self.train_labels += entry['labels']
#
#             self.train_data = np.concatenate(self.train_data)
#             self.train_data = self.train_data.reshape((50000, 3, 32, 32))
#             self.train_data = self.train_data.transpose((0, 2, 3, 1))
#
#             if self.validation_split:
#                 self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(self.train_data, self.train_labels, random_state)
#                 self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp', self.imb_factor)
#                 self.val_data = self.val_data.transpose((0, 3, 1, 2)).astype(np.uint8)  # Transpose and convert to uint8
#
#             self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 10, 'exp', self.imb_factor)
#
#             self.train_data = torch.from_numpy(self.train_data).byte()
#             self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
#
#             if partial_rate != 0.0:
#                 y = binarize_class(self.train_labels)
#                 self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
#             else:
#                 self.train_final_labels = binarize_class(self.train_labels).float()
#         else:
#             f = self.test_list[0][0]
#             file = os.path.join(self.root, self.base_folder, f)
#             with open(file, 'rb') as fo:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#             self.test_data = entry['data']
#             self.test_labels = entry['labels']
#             self.test_data = self.test_data.reshape((10000, 3, 32, 32))
#             self.test_data = self.test_data.transpose((0, 2, 3, 1))
#
#             self.test_data = torch.from_numpy(self.test_data).byte()
#             self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
#
#     def create_validation_split(self, data, labels, random_state):
#         train_data, val_data, train_labels, val_labels = train_test_split(
#             data, labels, test_size=0.2, stratify=labels, random_state=random_state
#         )
#         return train_data, val_data, train_labels, val_labels
#
#     def gen_imbalanced_data(self, data, targets, num_classes, imb_type='exp', imb_factor=0.01):
#         img_max = len(data) / num_classes
#         img_num_per_cls = self.get_img_num_per_cls(num_classes, imb_type, imb_factor, img_max)
#
#         new_data = []
#         new_targets = []
#         targets_np = np.array(targets, dtype=np.int64)
#         classes = np.unique(targets_np)
#
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             new_data.append(data[selec_idx, ...])
#             new_targets.extend([the_class] * the_img_num)
#
#         new_data = np.vstack(new_data)
#         new_targets = np.array(new_targets)
#
#         return new_data, new_targets
#
#     def get_img_num_per_cls(self, num_classes, imb_type, imb_factor, img_max):
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(num_classes):
#                 num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls = [int(img_max)] * num_classes
#         return img_num_per_cls
#
#     def __getitem__(self, index):
#         if self.train:
#             if self.validation_split and index >= len(self.train_data):
#                 index -= len(self.train_data)
#                 img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
#             else:
#                 img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
#         else:
#             img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
#
#         if isinstance(img, torch.Tensor):
#             img = img.numpy().astype(np.uint8)
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, true, index
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data) + (len(self.val_data) if self.validation_split else 0)
#         else:
#             return len(self.test_data)
#
#     def _check_integrity(self):
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True
#
#     def download(self):
#         import tarfile
#
#         if self._check_integrity():
#             return
#
#         download_url(self.url, self.root, self.filename, self.tgz_md5)
#
#         cwd = os.getcwd()
#         tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
#         os.chdir(self.root)
#         tar.extractall()
#         tar.close()
#         os.chdir(cwd)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         tmp = 'train' if self.train is True else 'test'
#         fmt_str += '    Split: {}\n'.format(tmp)
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str




from PIL import Image
import os
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url
from sklearn.model_selection import train_test_split

# class cifar10(data.Dataset):
#     base_folder = 'cifar-10-batches-py'
#     url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#     filename = 'cifar-10-python.tar.gz'
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#
#     def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
#                  partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.01, validation_split=True, imbalance=True):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train_or_not
#         self.dataset = 'cifar10'
#         self.imb_factor = imb_factor
#         self.validation_split = validation_split
#         self.imbalance = imbalance
#
#         if download:
#             self.download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         if self.train:
#             self.train_data = []
#             self.train_labels = []
#             for fentry in self.train_list:
#                 f = fentry[0]
#                 file = os.path.join(self.root, self.base_folder, f)
#                 with open(file, 'rb') as fo:
#                     entry = pickle.load(fo, encoding='latin1')
#                 self.train_data.append(entry['data'])
#                 self.train_labels += entry['labels']
#
#             self.train_data = np.concatenate(self.train_data)
#             self.train_data = self.train_data.reshape((50000, 3, 32, 32))
#             self.train_data = self.train_data.transpose((0, 2, 3, 1))
#
#             if self.validation_split:
#                 self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(self.train_data, self.train_labels, random_state)
#                 if self.imbalance:
#                     self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp', self.imb_factor)
#                 self.val_data = self.val_data.transpose((0, 3, 1, 2)).astype(np.uint8)  # Transpose and convert to uint8
#
#             if self.imbalance:
#                 self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 10, 'exp', self.imb_factor)
#
#             self.train_data = torch.from_numpy(self.train_data).byte()
#             self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
#
#             if partial_rate != 0.0:
#                 y = binarize_class(self.train_labels)
#                 self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
#             else:
#                 self.train_final_labels = binarize_class(self.train_labels).float()
#         else:
#             f = self.test_list[0][0]
#             file = os.path.join(self.root, self.base_folder, f)
#             with open(file, 'rb') as fo:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#             self.test_data = entry['data']
#             self.test_labels = entry['labels']
#             self.test_data = self.test_data.reshape((10000, 3, 32, 32))
#             self.test_data = self.test_data.transpose((0, 2, 3, 1))
#
#             self.test_data = torch.from_numpy(self.test_data).byte()
#             self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
#
#     def create_validation_split(self, data, labels, random_state):
#         train_data, val_data, train_labels, val_labels = train_test_split(
#             data, labels, test_size=0.2, stratify=labels, random_state=random_state
#         )
#         return train_data, val_data, train_labels, val_labels
#
#     def gen_imbalanced_data(self, data, targets, num_classes, imb_type='exp', imb_factor=0.01):
#         img_max = len(data) / num_classes
#         img_num_per_cls = self.get_img_num_per_cls(num_classes, imb_type, imb_factor, img_max)
#
#         new_data = []
#         new_targets = []
#         targets_np = np.array(targets, dtype=np.int64)
#         classes = np.unique(targets_np)
#
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             new_data.append(data[selec_idx, ...])
#             new_targets.extend([the_class] * the_img_num)
#
#         new_data = np.vstack(new_data)
#         new_targets = np.array(new_targets)
#
#         return new_data, new_targets
#
#     def get_img_num_per_cls(self, num_classes, imb_type, imb_factor, img_max):
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(num_classes):
#                 num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(num_classes // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls = [int(img_max)] * num_classes
#         return img_num_per_cls
#
#     def __getitem__(self, index):
#         if self.train:
#             if self.validation_split and index >= len(self.train_data):
#                 index -= len(self.train_data)
#                 img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
#             else:
#                 img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
#         else:
#             img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]
#
#         print(f"Index: {index}, Img shape: {img.shape}, Target: {target}, True: {true}")
#
#         if isinstance(img, torch.Tensor):
#             img = img.numpy().astype(np.uint8)
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, true, index
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data) + (len(self.val_data) if self.validation_split else 0)
#         else:
#             return len(self.test_data)
#
#     def _check_integrity(self):
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True
#
#     def download(self):
#         import tarfile
#
#         if self._check_integrity():
#             return
#
#         download_url(self.url, self.root, self.filename, self.tgz_md5)
#
#         cwd = os.getcwd()
#         tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
#         os.chdir(self.root)
#         tar.extractall()
#         tar.close()
#         os.chdir(cwd)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         tmp = 'train' if self.train is True else 'test'
#         fmt_str += '    Split: {}\n'.format(tmp)
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


from PIL import Image
import os
import sys
import torch
import numpy as np
import pickle
import torch.utils.data as data
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url
from sklearn.model_selection import train_test_split

class cifar10(data.Dataset):
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

    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0, imb_factor=0.1, validation_split=True, imbalance=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.dataset = 'cifar10'
        self.imb_factor = imb_factor
        self.validation_split = validation_split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

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

            if self.validation_split:
                self.train_data, self.val_data, self.train_labels, self.val_labels = self.create_validation_split(self.train_data, self.train_labels, random_state)
                self.val_data, self.val_labels = self.gen_imbalanced_data(self.val_data, self.val_labels, 10, 'exp', self.imb_factor)
                self.val_data = self.val_data.transpose((0, 3, 1, 2)).astype(np.uint8)  # Transpose and convert to uint8

            self.train_data, self.train_labels = self.gen_imbalanced_data(self.train_data, self.train_labels, 10, 'exp', self.imb_factor)

            self.train_data = torch.from_numpy(self.train_data).byte()
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)

            if partial_rate != 0.0:
                y = binarize_class(self.train_labels)
                self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
            else:
                self.train_final_labels = binarize_class(self.train_labels).float()
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, 'rb') as fo:
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            self.test_labels = entry['labels']
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

            self.test_data = torch.from_numpy(self.test_data).byte()
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)

    def create_validation_split(self, data, labels, random_state):
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=random_state
        )
        return train_data, val_data, train_labels, val_labels

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
            if self.validation_split and index >= len(self.train_data):
                index -= len(self.train_data)
                img, target, true = self.val_data[index], self.val_labels[index], self.val_labels[index]
            else:
                img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
        else:
            img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]

        print(f"Index: {index}, Initial Img shape: {img.shape}, Target: {target}, True: {true}")

        if isinstance(img, torch.Tensor):
            img = img.numpy().astype(np.uint8)
        print(f"Index: {index}, Post Conversion Img shape: {img.shape}, Target: {target}, True: {true}")

        img = Image.fromarray(img)
        print(f"Index: {index}, Image type: {type(img)}")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        print(f"Index: {index}, Final Img shape: {img.shape}, Target: {target}, True: {true}")
        return img, target, true, index

    def __len__(self):
        if self.train:
            return len(self.train_data) + (len(self.val_data) if self.validation_split else 0)
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
