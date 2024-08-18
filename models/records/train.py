import argparse
import builtins
import os
import random
import shutil
import time
import warnings
from sklearn.utils import shuffle
import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from datetime import datetime
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
import pandas as pd
from models.resnet import *
from utils.utils_algo import *
from utils.utils_loss import CORR_loss, CORR_loss_RECORDS, CORR_loss_RECORDS_mixup
from utils.cifar100 import load_cifar100_imbalance
from utils.cifar10 import load_cifar10_imbalance
torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2023 paper "Long-Tailed Partial Label Learning via Dynamic Rebalancing"')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10_im', 'cifar100_im'], help='dataset name')
parser.add_argument('--exp_dir', default='experiment/CORR', type=str, help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'], help='network architecture')
parser.add_argument('-j', '--workers', default=32, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=800, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900', help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False, help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)', dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://localhost:10002', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch '
                                                                             'N processes per node, which has N GPUs. This is the '
                                                                             'fastest way to use PyTorch for either single node or '
                                                                             'multi node data parallel training')
parser.add_argument('--num_class', default=10, type=int, help='number of class')
parser.add_argument('--upd_start', default=1, type=int, help='Start Updating w')
parser.add_argument('--partial_rate', default=0.1, type=float, help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true', help='for CIFAR-100 fine-grained training')
parser.add_argument('--imb_factor', default=0.01, type=float, help='dataset imbalance rate')
parser.add_argument('--records', action='store_true', help='use RECORDS')
parser.add_argument('--m', default=0.9, type=float, help='momentum for RECORDS')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha for mixup')
parser.add_argument('--disable_imbalancing', action='store_true', help='Disable imbalancing')

def main():
    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list(map(int, iterations))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    model_path = 'RECORDS_' if args.records else ''
    if args.mixup:
        model_path += f'mixup_alpha_{args.alpha}_'

    model_path = model_path + f'ds_{args.dataset}_pr_{args.partial_rate}_lr_{args.lr}_ep_{args.epochs}_us_{args.upd_start}_arch_{args.arch}_heir_{args.hierarchical}_if{args.imb_factor}_sd{args.seed}'
    args.exp_dir = os.path.join(args.exp_dir, model_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.exp_dir, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # create model
    print(f"=> creating model '{args.arch}'")
    model = ResNet_s(name='resnet18', num_class=args.num_class, pretrained=False)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}' if args.gpu is not None else None
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.dataset == 'cifar100_im':
        train_loader, train_givenY, val_loader, train_sampler, test_loader, cls_num_list_true_label = load_cifar100_imbalance(partial_rate=args.partial_rate, batch_size=args.batch_size, hierarchical=args.hierarchical, imb_factor=args.imb_factor, con=True, shuffle=True, disable_imbalancing=args.disable_imbalancing)
    elif args.dataset == 'cifar10_im':
        train_loader, train_givenY, val_loader,train_sampler, test_loader, cls_num_list_true_label = load_cifar10_imbalance(partial_rate=args.partial_rate, batch_size=args.batch_size, hierarchical=args.hierarchical, imb_factor=args.imb_factor, con=True, shuffle=True, disable_imbalancing=args.disable_imbalancing)

    print('Calculating uniform targets...')
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.cuda()
    if args.records:
        if args.mixup:
            loss_fn = CORR_loss_RECORDS_mixup(confidence, m=args.m, mixup=args.alpha)
        else:
            loss_fn = CORR_loss_RECORDS(confidence, m=args.m)
    else:
        loss_fn = CORR_loss(confidence)

    logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2) if args.gpu == 0 else None

    print('\nStart Training\n')
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        start_upd = epoch >= args.upd_start
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)
        train(train_loader, model, loss_fn, optimizer, epoch, args, logger, start_upd)
        feat_mean = loss_fn.feat_mean if args.records else None
        val_acc = test(model, val_loader, args, epoch, logger, feat_mean, 'validation')
        acc_test = test(model, test_loader, args, epoch, logger, feat_mean, 'test')
        mmc = loss_fn.confidence.max(dim=1)[0].mean()

        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write(f'Epoch {epoch}: Val Acc {val_acc}, Test Acc {acc_test}, Best Val Acc {best_acc}. (lr {optimizer.param_groups[0]["lr"]})\n')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, is_best=is_best, filename=f'{args.exp_dir}/checkpoint.pth.tar', best_file_name=f'{args.exp_dir}/checkpoint_best.pth.tar')

def train(train_loader, model, loss_fn, optimizer, epoch, args, tb_logger, start_upd=False):
    """Train for one epoch on the training set."""
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, acc_cls, loss_cls_log], prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()
        if args.mixup:
            pseudo_label = loss_fn.confidence[index, :].clone().detach()
            l = max(np.random.beta(4, 4), 1 - np.random.beta(4, 4))
            idx = torch.randperm(X_w.size(0))
            X_w_rand, X_s_rand = X_w[idx], X_s[idx]
            pseudo_label_rand = pseudo_label[idx]
            X_w_mix = l * X_w + (1 - l) * X_w_rand
            X_s_mix = l * X_s + (1 - l) * X_s_rand
            pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand
            cls_out, feat = model(torch.cat((X_w, X_s, X_w_mix, X_s_mix), 0))
            batch_size = X_w.shape[0]
            cls_out_w, cls_out_s, cls_out_w_mix, cls_out_s_mix = torch.split(cls_out, batch_size, dim=0)
            feat_w, feat_s, _, _ = torch.split(feat, batch_size, dim=0)
        else:
            cls_out, feat = model(torch.cat((X_w, X_s), 0))
            batch_size = X_w.shape[0]
            cls_out_w, cls_out_s = torch.split(cls_out, batch_size, dim=0)
            feat_w, feat_s = torch.split(feat, batch_size, dim=0)

        if args.records:
            if args.mixup:
                loss_cls = loss_fn(cls_out_w, cls_out_s, cls_out_w_mix, cls_out_s_mix, feat_w, feat_s, model, index, pseudo_label_mix, start_upd)
            else:
                loss_cls = loss_fn(cls_out_w, cls_out_s, feat_w, feat_s, model, index, start_upd)
        else:
            loss_cls = loss_fn(cls_out_w, cls_out_s, index, start_upd)
        loss = loss_cls
        loss_cls_log.update(loss_cls.item())
        acc = accuracy(cls_out_w, Y_true)[0]
        acc_cls.update(acc[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)

def test(model, test_loader, args, epoch, tb_logger, feat_mean=None, phase='test'):
    """test on the validation or test set."""
    all_logits = []
    all_labels = []
    with torch.no_grad():
        print(f'==> Evaluation on {phase} set...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        if feat_mean is not None:
            bias = model.module.fc(feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, eval_only=True)
            if feat_mean is not None:
                outputs = outputs - torch.log(bias + 1e-9)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        if args.gpu == 0:
            logits_df = pd.DataFrame(np.vstack(all_logits))
            labels_df = pd.DataFrame(np.hstack(all_labels), columns=['label'])
            results_df = pd.concat([logits_df, labels_df], axis=1)
            results_df.to_csv(os.path.join(args.exp_dir, f'{phase}_results.csv'), index=False)

        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)
        acc_tensors /= args.world_size

        print(f'Accuracy is {acc_tensors[0]:.2f}% ({acc_tensors[1]:.2f}%)')
        if args.gpu == 0:
            tb_logger.log_value(f'Top1 Acc ({phase})', acc_tensors[0], epoch)
            tb_logger.log_value(f'Top5 Acc ({phase})', acc_tensors[1], epoch)
    return acc_tensors[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
