from model import SimpleNet
from torch import optim


def settings(base_lr=1e-3):
    model = SimpleNet()
    # optimizer = meg.optimizer.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # lr_scheduler = meg.optimizer.LRScheduler(optimizer)
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=1e-2, cycle_momentum=False)
    return model, optimizer, lr_scheduler
