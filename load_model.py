from model import SimpleNet
from torch import optim
import torch as meg
from pathlib import Path
from collections import OrderedDict


def settings(base_lr=1e-5, pretrained=None, cuda=False):
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    if cuda:
        model = model.cuda()
    if pretrained == 'torch_pretrained.ckp':
        pretrained = Path(pretrained)
        with pretrained.open("rb") as f:
            if cuda:
                states = meg.load(f)
            else:
                states = meg.load(f, map_location=meg.device('cpu'))
            new_states = OrderedDict()
            new_states = new_states.fromkeys(list(model.state_dict().keys()))
            cnt = 0
            vals = list(states.values())
            for key in new_states:
                new_states[key] = vals[cnt]
                cnt += 1
            model.load_state_dict(new_states)
            model.eval()
    elif pretrained == 'checkpoint.pth':
        model_ckpt = meg.load(pretrained)
        model.load_state_dict(model_ckpt['model'])
        optimizer.load_state_dict(model_ckpt['optimizer'])
        base_lr = model_ckpt['lr']
    # optimizer = meg.optimizer.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # lr_scheduler = meg.optimizer.LRScheduler(optimizer)
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=1e-3, cycle_momentum=False)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30)
    return model, optimizer, lr_scheduler


if __name__ == "__main__":
    pretrained = Path('torch_pretrained.ckp')
    model = SimpleNet()
    with pretrained.open("rb") as f:
        states = meg.load(f, map_location=meg.device('cpu'))
        new_states = OrderedDict()
        new_states = new_states.fromkeys(list(model.state_dict().keys()))
        cnt = 0
        vals = list(states.values())
        for key in new_states:
            new_states[key] = vals[cnt]
            cnt += 1
        model.load_state_dict(new_states)
        model.eval()
