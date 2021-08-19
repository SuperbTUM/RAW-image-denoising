from model_mod.model_modify import AnotherNet
from torch import optim


def settings(base_lr=1e-5, cuda=False):
    model = AnotherNet()
    if cuda:
        model = model.cuda()
    # optimizer = meg.optimizer.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # lr_scheduler = meg.optimizer.LRScheduler(optimizer)
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr=1e-3)
    return model, optimizer, lr_scheduler

