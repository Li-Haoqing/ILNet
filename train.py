import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
import time

from utils.Adan import Adan
from utils.data import SirstDataset, IRSTD1K_Dataset
from utils.lr_scheduler import adjust_learning_rate, create_lr_scheduler
from model.loss import SoftIoULoss, criterion
from model.metrics import IoUMetric, nIoUMetric, PD_FA
from model.ilnet import ILNet_S, ILNet_M, ILNet_L


# torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser(description='Implement of MY model')

    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')

    parser.add_argument('--amp', default=True, help='Use torch.cuda.amp for mixed precision training')

    args = parser.parse_args()
    return args


class Trainer(object):

    def __init__(self, args):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.train_set = SirstDataset(args, mode='train')
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.train_set = IRSTD1K_Dataset(args, mode='train')
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        else:
            NameError

        self.train_data_loader = Data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=8, pin_memory=True)
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size, num_workers=8, pin_memory=True)

        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.model = ILNet_L()
        elif args.mode == 'M':
            self.model = ILNet_M()
        elif args.mode == 'S':
            self.model = ILNet_S()
        else:
            NameError

        self.model = self.model.cuda()
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # self.criterion = SoftIoULoss()
        self.criterion = criterion

        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0)
        self.optimizer = Adan(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        self.lr_scheduler = create_lr_scheduler(self.optimizer, len(self.train_data_loader), args.epochs, warmup=True,
                                                warmup_epochs=args.warm_up_epochs)
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0
        self.best_PD = 0
        self.best_FA = 1
        self.PD_FA = PD_FA(args.img_size)

        if args.resume:
            folder_name = os.path.abspath(
                os.path.dirname(os.path.abspath(os.path.dirname(args.resume) + os.path.sep + "."))
                + os.path.sep + ".")
        else:
            folder_name = '%s_bs%s_lr%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                            args.batch_size, args.learning_rate)

        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.save_folder = ops.join('results_sirst/', folder_name)  # sirst
            self.save_pth = ops.join(self.save_folder, 'checkpoint')
            if not ops.exists('results_sirst'):
                os.mkdir('results_sirst')
            if not ops.exists(self.save_folder):
                os.mkdir(self.save_folder)
            if not ops.exists(self.save_pth):
                os.mkdir(self.save_pth)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.save_folder2 = ops.join('results_IRSTD-1k/', folder_name)  # IRSTD-1k
            self.save_pth2 = ops.join(self.save_folder2, 'checkpoint')
            if not ops.exists('results_IRSTD-1k'):
                os.mkdir('results_IRSTD-1k')
            if not ops.exists(self.save_folder2):
                os.mkdir(self.save_folder2)
            if not ops.exists(self.save_pth2):
                os.mkdir(self.save_pth2)

        # Tensorboard SummaryWriter
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.writer = SummaryWriter(log_dir=self.save_folder)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.model, next(iter(self.train_data_loader))[0].cuda())
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.writer = SummaryWriter(log_dir=self.save_folder2)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.model, next(iter(self.train_data_loader))[0].cuda())

        if self.train_set.__class__.__name__ == 'SirstDataset':
            print('folder: %s' % self.save_folder)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            print('folder: %s' % self.save_folder2)
        print('Args: %s' % args)

    def training(self, epoch):

        losses = []
        self.model.train()
        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            data, labels = data.cuda(), labels.cuda()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(data)
                loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()

            losses.append(loss.item())
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses)))
        # adjust_learning_rate(self.optimizer, epoch, args.epochs, args.learning_rate,
        #                      args.warm_up_epochs, 1e-6)

        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):

        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        eval_losses = []
        self.model.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.model(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.5f'
                                 % (epoch, np.mean(eval_losses), IoU, nIoU, Fa, Pd))

        pkl_name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f_Fa:%.8f_Pd:%.5f.pth' % (epoch, IoU, nIoU, Fa, Pd)
        save_file = {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "lr_scheduler": self.lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = self.scaler.state_dict()

        if self.train_set.__class__.__name__ == 'SirstDataset':
            save_pth = self.save_pth
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            save_pth = self.save_pth2
        if IoU > self.best_iou:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_nIoU = nIoU
        if Pd > self.best_PD:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_PD = Pd
        if Fa < self.best_FA:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_FA = Fa

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        self.writer.add_scalar('Eval/Pd', Pd, epoch)
        self.writer.add_scalar('Eval/Fa', Fa, epoch)
        self.writer.add_scalar('Best/Pd', self.best_PD, epoch)
        self.writer.add_scalar('Best/Fa', self.best_FA, epoch)


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs + 1):
        trainer.training(epoch)
        # if epoch > 5:
        #     trainer.validation(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f, best nIoU: %.5f, Best Pd: %.5f, best Fa: %.5f' %
          (trainer.best_iou, trainer.best_nIoU, trainer.best_PD, trainer.best_FA))
