import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from utils.randaugment import RandomAugment
from utils.utils_algo import generate_uniform_cv_candidate_labels
from utils.imbalance_cifar import IMBALANCECIFAR10


def load_cifar10_imbalance(partial_rate, batch_size, hierarchical=False, imb_type='exp', imb_factor=0.01, con=True,
                           test=False, shuffle=False, disable_imbalancing=False):
    """Load PLL version of CIFAR-10-LT dataset

    Args:
        partial_rate: Ambiguity q in PLL
        batch_size: batch size
        hierarchical (bool, optional): False for CIFAR10.
        imb_type (str, optional): Type of imbalance. Defaults to 'exp'.
        imb_factor (float, optional): Imbalance ratio: min_num / max_num. Defaults to 0.01.
        con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        test (bool, optional): Whether to return test loader. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the classes when generating the LT dataset. Defaults to False.
        disable_imbalancing (bool, optional): Whether to disable imbalancing. Defaults to False.

    """
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # Load the full training dataset
    full_train_dataset = dsets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)

    # Split the full training dataset into train and validation sets
    num_train = len(full_train_dataset)
    split = int(np.floor(0.8 * num_train))
    indices = list(range(num_train))
    train_idx, val_idx = indices[:split], indices[split:]

    # Creating validation loader similar to test loader
    val_data, val_labels = full_train_dataset.data[split:], full_train_dataset.targets[split:]
    val_dataset = dsets.CIFAR10(root='./data', train=True, transform=test_transform)
    val_dataset.data = val_data
    val_dataset.targets = val_labels
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if disable_imbalancing:
        # Use the balanced dataset for training
        train_data, train_labels = full_train_dataset.data[:split], torch.Tensor(
            full_train_dataset.targets[:split]).long()
    else:
        # Create the imbalanced training set using the first 80% of the data
        temp_train = IMBALANCECIFAR10(root='./data', train=True, download=True, imb_type=imb_type,
                                      imb_factor=imb_factor, shuffle=shuffle)
        train_data, train_labels = temp_train.data[:split], torch.Tensor(temp_train.targets[:split]).long()

    # Generate partial labels
    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)

    # Validate partial labels
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())

    train_dataset = CIFAR10_Augmentention(train_data, partialY.float(), train_labels.float(), con=con)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=4,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)

    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=4,
                             sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))

    num_classes = len(np.unique(train_labels))
    assert num_classes == 10
    cls_num_list_true_label = [0] * num_classes
    for label in train_labels:
        cls_num_list_true_label[label] += 1

    if test:
        return test_loader, cls_num_list_true_label

    return train_loader,partialY, val_loader, train_sampler, test_loader, cls_num_list_true_label
    #train_loader, train_givenY, val_loader, train_sampler, test_loader, cls_num_list_true_label


class CIFAR10_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, con=True):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.con = con

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        if self.con:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]

            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_image_w = self.weak_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index
