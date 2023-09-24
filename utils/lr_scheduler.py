# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import math


def adjust_learning_rate(optimizer, epoch, epochs, lr, warm_up_epochs=0, min_lr=0):
    if epoch < warm_up_epochs:
        cur_lr = lr * epoch / warm_up_epochs
    else:
        cur_lr = pow(1 - float(epoch - warm_up_epochs) / (epochs - warm_up_epochs + 1), 0.9) \
                 * (lr - min_lr) + min_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
