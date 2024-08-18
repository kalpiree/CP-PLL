
import torch
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np
import time

from datasets.CUB200 import CUB200
from datasets.cifar100 import cifar100
from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from cifar_models import convnet, resnet, resnet_
from datasets.mnist import mnist
from datasets.fashion import fashion
from datasets.kmnist import kmnist
from datasets.CIFAR10_ import CIFAR10  # Updated import
from datasets.CIFAR100_ import IMBALANCECIFAR100
from datasets.CUB200_ import IMBALANCECUB200

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np

from datasets.CUB200 import CUB200
from datasets.cifar100 import cifar100
from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from cifar_models import convnet, resnet, resnet_
from datasets.mnist import mnist
from datasets.fashion import fashion
from datasets.kmnist import kmnist
from datasets.cifar10 import cifar10

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
    prog='PRODEN demo file.',
    usage='Demo with partial labels.',
    description='A simple demo file with MNIST dataset.',
    epilog='end',
    add_help=True)

parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'cifar100', 'cub200'], required=False)
parser.add_argument('-model', help='model name', type=str, default='linear',
                    choices=['linear', 'mlp', 'convnet', 'resnet','resnet_'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)
parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial',
                    choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)
parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
parser.add_argument('-validation_split', help='use validation split', action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar10(root='./cifar10/',
                            download=True,
                            train_or_not=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                               (0.2023, 0.1994, 0.2010)), ]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate,
                            validation_split=args.validation_split
                            )
    test_dataset = cifar10(root='./cifar10/',
                           download=True,
                           train_or_not=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                              (0.2023, 0.1994, 0.2010)), ]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate
                           )

# Learning rate plan
lr_plan = [args.lr] * args.ep
for i in range(args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]

# Result directory
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file_train = os.path.join(save_dir, 'train_' + args.partial_type + '_' + str(args.partial_rate) + '.txt')
save_file_val = os.path.join(save_dir, 'val_' + args.partial_type + '_' + str(args.partial_rate) + '.txt')
save_file_test = os.path.join(save_dir, 'test_' + args.partial_type + '_' + str(args.partial_rate) + '.txt')
save_scores_val = os.path.join(save_dir, 'val_scores_' + args.partial_type + '_' + str(args.partial_rate) + '.npy')
save_scores_test = os.path.join(save_dir, 'test_scores_' + args.partial_type + '_' + str(args.partial_rate) + '.npy')

# Calculate accuracy and save probabilistic scores
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    all_scores = []
    all_labels = []
    for images, _, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        output1 = model(images)
        output = F.softmax(output1, dim=1)
        all_scores.append(output.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return acc, all_scores, all_labels

def main():
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.nw,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)

    val_loader = None
    if args.validation_split:
        val_indices = range(len(train_dataset.train_data), len(train_dataset))
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.bs,
                                                 num_workers=args.nw,
                                                 drop_last=False,
                                                 shuffle=False)

    if args.model == 'linear':
        net = linear(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        net = mlp(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = resnet(depth=32, n_outputs=num_classes)
    elif args.model == 'resnet_':
        net = resnet(depth=50, n_outputs=num_classes)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(args.ep):
        net.train()
        adjust_learning_rate(optimizer, epoch)

        for images, labels, trues, indexes in train_loader:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            output = net(images)

            loss, new_label = partial_loss(output, labels, trues)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()

        train_acc, _, _ = evaluate(train_loader, net)
        test_acc, test_scores, test_labels = evaluate(test_loader, net)
        val_acc, val_scores, val_labels = evaluate(val_loader, net) if val_loader else (None, None, None)

        with open(save_file_train, 'a') as file:
            file.write(f"{epoch}: Training Acc.: {train_acc:.4f}\n")

        with open(save_file_test, 'a') as file:
            file.write(f"{epoch}: Test Acc.: {test_acc:.4f}\n")

        if val_acc is not None:
            with open(save_file_val, 'a') as file:
                file.write(f"{epoch}: Validation Acc.: {val_acc:.4f}\n")

        if epoch == args.ep - 1:
            np.save(save_scores_test, {'scores': test_scores, 'labels': test_labels})
            if val_scores is not None:
                np.save(save_scores_val, {'scores': val_scores, 'labels': val_labels})

if __name__ == '__main__':
    main()



# if args.ds == 'mnist':
#     num_features = 28 * 28
#     num_classes = 10
#     num_training = 60000
#     train_dataset = mnist(root='/Users/nitinbisht/Downloads/PRODEN-master/mnist/',
#                           download=False,
#                           train_or_not=True,
#                           transform=transforms.Compose(
#                               [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                           partial_type=args.partial_type,
#                           partial_rate=args.partial_rate
#                           )
#     test_dataset = mnist(root='/Users/nitinbisht/Downloads/PRODEN-master/mnist/',
#                          download=False,
#                          train_or_not=False,
#                          transform=transforms.Compose(
#                              [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                          partial_type=args.partial_type,
#                          partial_rate=args.partial_rate
#                          )
#
# if args.ds == 'fashion':
#     num_features = 28 * 28
#     num_classes = 10
#     num_training = 60000
#     train_dataset = fashion(root='./fashion/',
#                             download=True,
#                             train_or_not=True,
#                             transform=transforms.Compose(
#                                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                             partial_type=args.partial_type,
#                             partial_rate=args.partial_rate
#                             )
#     test_dataset = fashion(root='./fashionmnist/',
#                            download=True,
#                            train_or_not=False,
#                            transform=transforms.Compose(
#                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                            partial_type=args.partial_type,
#                            partial_rate=args.partial_rate
#                            )
#
# if args.ds == 'kmnist':
#     num_features = 28 * 28
#     num_classes = 10
#     num_training = 60000
#     train_dataset = kmnist(root='./kmnist/',
#                            download=True,
#                            train_or_not=True,
#                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
#                            partial_type=args.partial_type,
#                            partial_rate=args.partial_rate
#                            )
#     test_dataset = kmnist(root='./kmnist/',
#                           download=True,
#                           train_or_not=False,
#                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
#                           partial_type=args.partial_type,
#                           partial_rate=args.partial_rate
#                           )
# if args.ds == 'cifar10':
#     input_channels = 3
#     num_classes = 10
#     dropout_rate = 0.25
#     num_training = 50000
#     train_dataset = cifar10(root='./cifar10/',
#                             download=True,
#                             train_or_not=True,
#                             transform=transforms.Compose([transforms.ToTensor(),
#                                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                                (0.2023, 0.1994, 0.2010)), ]),
#                             partial_type=args.partial_type,
#                             partial_rate=args.partial_rate,
#
#                             )
#     test_dataset = cifar10(root='./cifar10/',
#                            download=True,
#                            train_or_not=False,
#                            transform=transforms.Compose([transforms.ToTensor(),
#                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                               (0.2023, 0.1994, 0.2010)), ]),
#                            partial_type=args.partial_type,
#                            partial_rate=args.partial_rate,
#
#                            )

#if args.ds == 'cifar100':
#     input_channels = 3
#     num_classes = 100
#     dropout_rate = 0.25
#     num_training = 50000
#
#     train_dataset = cifar100(root='./cifar100/',
#                              download=True,
#                              train_or_not=True,
#                              validation_split=False,
#                              transform=transforms.Compose([transforms.ToTensor(),
#                                                            transforms.Normalize((0.5071, 0.4867, 0.4408),
#                                                                                 (0.2675, 0.2565, 0.2761)), ]),
#                              partial_type=args.partial_type,
#                              partial_rate=args.partial_rate
#                              )
#     test_dataset = cifar100(root='./cifar100/',
#                             download=True,
#                             train_or_not=False,
#                             transform=transforms.Compose([transforms.ToTensor(),
#                                                           transforms.Normalize((0.5071, 0.4867, 0.4408),
#                                                                                (0.2675, 0.2565, 0.2761)), ]),
#                             partial_type=args.partial_type,
#                             partial_rate=args.partial_rate
#                             )
# if args.ds == 'cub200':
#     input_channels = 3
#     num_classes = 200
#     dropout_rate = 0.25
#     num_training = 5994  # Update with actual training samples
#     train_dataset = CUB200(root='./cub200/',
#                            download=True,
#                            train_or_not=True,
#                            transform=transforms.Compose([transforms.Resize((224, 224)),
#                                                          transforms.ToTensor(),
#                                                          transforms.Normalize((0.485, 0.456, 0.406),
#                                                                               (0.229, 0.224, 0.225))]),
#                            partial_type=args.partial_type,
#                            partial_rate=args.partial_rate
#                            )
#     test_dataset = CUB200(root='./cub200/',
#                           download=True,
#                           train_or_not=False,
#                           transform=transforms.Compose([transforms.Resize((224, 224)),
#                                                         transforms.ToTensor(),
#                                                         transforms.Normalize((0.485, 0.456, 0.406),
#                                                                              (0.229, 0.224, 0.225))]),
#                           partial_type=args.partial_type,
#                           partial_rate=args.partial_rate
#                           )


# import os
# import os.path
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import argparse
# import numpy as np
# import time

from datasets.CUB200 import CUB200
from datasets.cifar100 import cifar100
# from utils.utils_loss import partial_loss
# from utils.models import linear, mlp
# from cifar_models import convnet, resnet, resnet_
# from datasets.mnist import mnist
# from datasets.fashion import fashion
# from datasets.kmnist import kmnist
# from datasets.cifar10_lt_ import cifar10
#
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
#
# parser = argparse.ArgumentParser(
#     prog='PRODEN demo file.',
#     usage='Demo with partial labels.',
#     description='A simple demo file with MNIST dataset.',
#     epilog='end',
#     add_help=True)
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
# parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
# parser.add_argument('-bs', help='batch size', type=int, default=256)
# parser.add_argument('-ep', help='number of epochs', type=int, default=500)
# parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist',
#                     choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'cifar100', 'cub200', 'cifar10_lt'], required=False)
# parser.add_argument('-model', help='model name', type=str, default='linear',
#                     choices=['linear', 'mlp', 'convnet', 'resnet', 'resnet_'], required=False)
# parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
# parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)
#
# parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial',
#                     choices=['binomial', 'pair'])
# parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)
#
# parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
# parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
# parser.add_argument('-imbalance', help='apply imbalanced data', type=bool, default=True, required=False)
#
# args = parser.parse_args()
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Load dataset
#
#
#
# if args.ds == 'cifar10_lt':
#     input_channels = 3
#     num_classes = 10
#     dropout_rate = 0.25
#     num_training = 50000
#     train_dataset = cifar10(root='./cifar10_lt/',
#                             download=True,
#                             train_or_not=True,
#                             validation_split=True,
#                             transform=transforms.Compose([transforms.ToTensor(),
#                                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                                (0.2023, 0.1994, 0.2010)), ]),
#                             partial_type=args.partial_type,
#                             partial_rate=args.partial_rate,
#                             imbalance=args.imbalance
#                             )
#     test_dataset = cifar10(root='./cifar10_lt/',
#                            download=True,
#                            train_or_not=False,
#                            transform=transforms.Compose([transforms.ToTensor(),
#                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                               (0.2023, 0.1994, 0.2010)), ]),
#                            partial_type=args.partial_type,
#                            partial_rate=args.partial_rate,
#                            imbalance=args.imbalance
#                            )
#
#
#
# # Learning rate
# lr_plan = [args.lr] * args.ep
# for i in range(0, args.ep):
#     lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)
#
#
# def adjust_learning_rate(optimizer, epoch):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr_plan[epoch]
#
#
# # Result directory
# save_dir = './' + args.dir
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_file = os.path.join(save_dir, (args.partial_type + '_' + str(args.partial_rate) + '.txt'))
# val_scores_file = os.path.join(save_dir, 'val_scores.txt')
# test_scores_file = os.path.join(save_dir, 'test_scores.txt')
#
#
# # Calculate accuracy
#
#
# def evaluate(loader, model):
#     model.eval()
#     correct = 0
#     total = 0
#     all_scores = []
#     all_labels = []
#     with torch.no_grad():
#         for images, labels, trues, indexes in loader:  # Ensure consistent unpacking
#             images = images.to(device)
#             labels = labels.to(device)
#
#             output = model(images)
#             output = F.softmax(output, dim=1)
#             print(f"Output shape: {output.shape}, Labels shape: {labels.shape}")
#             _, pred = torch.max(output, 1)
#             print(f"Pred shape: {pred.shape}, Labels shape: {labels.shape}")
#
#             # Convert labels to the same shape as predictions
#             if labels.ndim == 2 and labels.shape[1] > 1:
#                 labels = labels.argmax(dim=1)
#
#             print(f"Converted Labels shape: {labels.shape}")
#             total += images.size(0)
#             correct += (pred == labels).sum().item()
#             all_scores.extend(output.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     acc = 100 * float(correct) / float(total)
#     return acc, all_scores, all_labels
#
#
# def save_scores_and_labels(scores, labels, file_path):
#     with open(file_path, 'w') as f:
#         for score, label in zip(scores, labels):
#             f.write(f'{label}, {score.tolist()}\n')
#
#
# def main():
#     # Load dataset
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=args.bs,
#                                                num_workers=args.nw,
#                                                drop_last=True,
#                                                shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                               batch_size=args.bs,
#                                               num_workers=args.nw,
#                                               drop_last=False,
#                                               shuffle=False)
#
#     # Create validation loader if validation_split is enabled
#     if hasattr(train_dataset, 'val_data') and train_dataset.validation_split:
#         val_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#         val_dataset = torch.utils.data.TensorDataset(
#             torch.from_numpy(train_dataset.val_data).permute(0, 3, 1, 2).float(),  # Transpose to [3, 32, 32]
#             torch.tensor(train_dataset.val_labels, dtype=torch.long),
#             torch.tensor(train_dataset.val_labels, dtype=torch.long),
#             torch.arange(len(train_dataset.val_labels)))
#         val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                                  batch_size=args.bs,
#                                                  num_workers=args.nw,
#                                                  drop_last=False,
#                                                  shuffle=False)
#     else:
#         val_loader = None
#
#     # Build model
#     # if args.model == 'linear':
#     #     net = linear(n_inputs=num_features, n_outputs=num_classes)
#     # elif args.model == 'mlp':
#     #     net = mlp(n_inputs=num_features, n_outputs=num_classes)
#     if args.model == 'convnet':
#         net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
#     elif args.model == 'resnet':
#         net = resnet(depth=32, n_outputs=num_classes)
#     elif args.model == 'resnet_':
#         net = resnet(depth=50, n_outputs=num_classes)
#     net.to(device)
#     print(net.parameters)
#
#     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
#
#     for epoch in range(0, 2):
#         net.train()
#         adjust_learning_rate(optimizer, epoch)
#
#         for i, (images, labels, trues, indexes) in enumerate(train_loader):
#             images = Variable(images).to(device)
#             labels = Variable(labels).to(device)
#             trues = trues.to(device)
#             print(f"Epoch: {epoch}, Batch: {i}, Images shape: {images.shape}, Labels shape: {labels.shape}")
#             output = net(images)
#             print(f"Output shape: {output.shape}")
#
#             loss, new_label = partial_loss(output, labels, trues)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Update weights
#             for j, k in enumerate(indexes):
#                 train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()
#
#         # Evaluate model
#         train_acc, _, _ = evaluate(train_loader, net)
#         if val_loader:
#             val_acc, val_scores, val_labels = evaluate(val_loader, net)
#         else:
#             val_acc, val_scores, val_labels = 0, [], []
#         test_acc, test_scores, test_labels = evaluate(test_loader, net)
#
#         with open(save_file, 'a') as file:
#             file.write(f'{int(epoch)}: Training Acc.: {round(train_acc, 4)} , Validation Acc.: {round(val_acc, 4)}, '
#                        f'Test Acc.: {round(test_acc, 4)}\n')
#
#     if val_loader:
#         save_scores_and_labels(val_scores, val_labels, val_scores_file)
#     save_scores_and_labels(test_scores, test_labels, test_scores_file)
#
#
# if __name__ == '__main__':
#     main()