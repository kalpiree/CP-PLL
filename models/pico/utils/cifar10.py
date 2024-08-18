import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.utils.data as data
from torchvision.transforms import transforms

from utils.cifar10_lt import cifar10
from utils.randaugment import RandomAugment
from utils.utils_algo import generate_uniform_cv_candidate_labels
from torch.utils.data import Dataset, DataLoader


import torch
from torchvision.transforms import functional as F

class PermuteTransform:
    def __call__(self, x):
        return x.permute(2, 0, 1)
    

def load_cifar10(partial_rate, batch_size):
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    temp_train = cifar10(root='./cifar10_lt/',
                         download=True,
                         train_or_not=True,
                         val_or_not= False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010)), ]),
                         imb_factor=0.005,
                         imbalance=False
                         )
    val_dataset = cifar10(root='./cifar10_lt/',
                          download=True,
                          train_or_not=False,
                          val_or_not=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                             (0.2023, 0.1994, 0.2010)), ]),
                          imb_factor=0.005,
                          imbalance=False
                          )
    test_dataset = cifar10(root='./cifar10_lt/',
                           download=True,
                           train_or_not=False,
                           val_or_not= False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                              (0.2023, 0.1994, 0.2010)), ]),
                           # partial_type=args.partial_type,
                           # partial_rate=args.partial_rate,
                           imb_factor=0.005,
                           imbalance=False
                           )

    # temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data = temp_train.train_data
    #labels = torch.Tensor(temp_train.train_labels).long()
    labels = torch.tensor(temp_train.train_labels, dtype=torch.float).long()
    print(f"Initial number of images: {len(data)}, Initial number of labels: {len(labels)}")
    # data, labels = temp_train.img, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    # test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size * 4, shuffle=False,
                                             num_workers=4,
                                             sampler=torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                                                     shuffle=False))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False,
                                              num_workers=4,
                                              sampler=torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                                                      shuffle=False))
    # set test dataloader

    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    print(f"Partial label shape: {partialY.shape}")
    # generate partial labels
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float())
    print(f"Training dataset length: {len(partial_matrix_dataset)}")
    #print(f"Initial number of images: {len(data)}, Initial number of labels: {len(labels)}")
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              drop_last=True)
    return partial_matrix_train_loader, partialY, train_sampler, test_loader, val_loader


class CIFAR10_Augmentention(data.Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                PermuteTransform(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        self.strong_transform = transforms.Compose(
            [
                PermuteTransform(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
            )
        #print(f"Dataset initialized with {len(images)} images")
        print(f"Dataset initialized with {len(images)} images and {len(true_labels)} labels")

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        if index >= len(self.images):
            print(f"Index {index} is out of bounds for dataset with length {len(self.images)}")
        image = self.images[index]
        #print(f"Accessing image at index {index}, shape: {image.shape}")  # Check the shape here
        #image = self.images[index]
        #print(f"Image shape before transform: {image.shape}")  # Check the shape here
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image_w, each_image_s, each_label, each_true_label, index



