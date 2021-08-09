import rawpy
from load_model import settings
from utils import *


def get_rgb(path):
    raw = rawpy.imread(path)
    rgb = raw.postprocess(use_camera_wb=True)
    return rgb


def preprocess(rgb):
    rgb = rgb.transpose(2, 0, 1)
    rgb = norm(rgb)
    return rgb


def train(rgbs):
    train, gt = rgbs
    train = preprocess(train)
    gt = preprocess(gt)
    model, optimizer, lr_scheduler = settings()
