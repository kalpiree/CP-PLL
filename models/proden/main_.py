import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

from cifar_models import convnet, resnet
from datasets.cifar10_lt import cifar10
from datasets.cifar100_lt import cifar100
from utils.utils_loss import partial_loss

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    prog='PRODEN demo file.',
    usage='Demo with partial labels.',
    description='A simple demo file with MNIST dataset.',
    epilog='end',
    add_help=True)

parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=200)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'cifar100', 'cub200', 'cifar10_lt',
                             'cifar100_lt'], required=False)
parser.add_argument('-model', help='model name', type=str, default='linear',
                    choices=['linear', 'mlp', 'convnet', 'resnet', 'resnet_'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial',
                    choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)

parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
#parser.add_argument('-imbalance', help='apply imbalanced data', type=bool)#, default=True, required=False)
parser.add_argument('-imbalance', help='apply imbalanced data', type=str2bool, default=False, required=False)
parser.add_argument('-imb_rate', help='imbalance_rate', type=float, default=0.1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print arguments to verify
# print("Arguments received:")
# print(args)
# Load dataset
if args.ds == 'cifar10_lt':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 40000  # Updated to reflect the actual number of training samples
    train_dataset = cifar10(root='./cifar10_lt/',
                            download=True,
                            train_or_not=True,
                            val_or_not=False,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                               (0.2023, 0.1994, 0.2010)), ]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate,
                            imbalance=args.imbalance,
                            imb_factor=args.imb_rate
                            )
    val_dataset = cifar10(root='./cifar10_lt/',
                          download=True,
                          train_or_not=False,
                          val_or_not=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                             (0.2023, 0.1994, 0.2010)), ]),
                          partial_type=args.partial_type,
                          partial_rate=args.partial_rate,
                          imbalance=args.imbalance,
                          imb_factor=args.imb_rate
                          )
    test_dataset = cifar10(root='./cifar10_lt/',
                           download=True,
                           train_or_not=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                              (0.2023, 0.1994, 0.2010)), ]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate,
                           imbalance=args.imbalance,
                           imb_factor=args.imb_rate

                           )

elif args.ds == 'cifar100_lt':
    input_channels = 3
    num_classes = 100
    dropout_rate = 0.25
    num_training = 40000

    train_dataset = cifar100(root='./cifar100/',
                             download=True,
                             train_or_not=True,
                             val_or_not=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                                (0.2675, 0.2565, 0.2761)), ]),
                             partial_type=args.partial_type,
                             partial_rate=args.partial_rate,
                             imbalance=args.imbalance
                             )

    val_dataset = cifar100(root='./cifar100/',
                           download=True,
                           train_or_not=False,
                           val_or_not=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                              (0.2675, 0.2565, 0.2761)), ]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate,
                           imbalance=args.imbalance
                           )
    test_dataset = cifar100(root='./cifar100/',
                            download=True,
                            train_or_not=False,
                            val_or_not=False,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                               (0.2675, 0.2565, 0.2761)), ]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate,
                            imbalance=args.imbalance)

# Learning rate
lr_plan = [args.lr] * args.ep
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# # Result directory
# save_dir = './' + args.dir
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# # save_file = os.path.join(save_dir, (args.partial_type + '_' + str(args.partial_rate) + '.txt'))
# # save_res = os.path.join(save_dir, (
# #             args.partial_type + '_dataset_' + str(args.ds) + '_imbalance_' + str(args.imbalance) + '_'
# #             + str(args.partial_rate) + '.txt'))
# # save_file = os.path.join(save_res, (args.partial_type + '_' + str(args.partial_rate) + '.txt'))
# save_res = os.path.join(save_dir, f'{args.partial_type}_dataset_{args.ds}_imbalance_{args.imbalance}_{args.partial_rate}')
# save_file = os.path.join(save_res, f'{args.partial_type}_{args.partial_rate}.txt')
# val_scores_file = os.path.join(save_res, 'val_scores.txt')
# test_scores_file = os.path.join(save_res, 'test_scores.txt')

save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_res = os.path.join(save_dir, f'{args.partial_type}_dataset_{args.ds}_imbalance_{args.imbalance}_{args.partial_rate}')
if not os.path.exists(save_res):
    os.makedirs(save_res)
save_file = os.path.join(save_res, f'{args.partial_type}_{args.partial_rate}.txt')
val_scores_file = os.path.join(save_res, 'val_scores.txt')
test_scores_file = os.path.join(save_res, 'test_scores.txt')



# Calculate accuracy
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels, trues, indexes in loader:  # Ensure consistent unpacking
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            output = F.softmax(output, dim=1)
            # print(f"Output shape: {output.shape}, Labels shape: {labels.shape}")
            _, pred = torch.max(output, 1)
            # print(f"Pred shape: {pred.shape}, Labels shape: {labels.shape}")

            # Convert labels to the same shape as predictions
            if labels.ndim == 2 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)

            # print(f"Converted Labels shape: {labels.shape}")
            total += images.size(0)
            correct += (pred == labels).sum().item()
            all_scores.extend(output.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * float(correct) / float(total)
    return acc, all_scores, all_labels


# def save_scores_and_labels(scores, labels, file_path):
#     with open(file_path, 'w') as f:
#         for score, label in zip(scores, labels):
#             f.write(f'{label}, {score.tolist()}\n')

def save_scores_and_labels(scores, labels, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    with open(file_path, 'w') as f:
        for score, label in zip(scores, labels):
            f.write(f'{label}, {score.tolist()}\n')
def main():
    # Load dataset
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

    # Create validation loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.bs,
                                             num_workers=args.nw,
                                             drop_last=False,
                                             shuffle=False)

    if args.model == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = resnet(depth=32, n_outputs=num_classes)
    elif args.model == 'resnet_':
        net = resnet(depth=50, n_outputs=num_classes)
    net.to(device)
    print(list(net.parameters()))

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(0, args.ep):
        net.train()
        adjust_learning_rate(optimizer, epoch)

        for i, (images, labels, trues, indexes) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            # print(f"Epoch: {epoch}, Batch: {i}, Images shape: {images.shape}, Labels shape: {labels.shape}")
            output = net(images)
            # print(f"Output shape: {output.shape}")

            loss, new_label = partial_loss(output, labels, trues)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update weights
            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()

        # Evaluate model
        train_acc, _, _ = evaluate(train_loader, net)
        val_acc, val_scores, val_labels = evaluate(val_loader, net)
        test_acc, test_scores, test_labels = evaluate(test_loader, net)

        with open(save_file, 'a') as file:
            file.write(f'{int(epoch)}: Training Acc.: {round(train_acc, 4)} , Validation Acc.: {round(val_acc, 4)}, '
                       f'Test Acc.: {round(test_acc, 4)}\n')

        save_scores_and_labels(val_scores, val_labels, val_scores_file)
        save_scores_and_labels(test_scores, test_labels, test_scores_file)


if __name__ == '__main__':
    main()
