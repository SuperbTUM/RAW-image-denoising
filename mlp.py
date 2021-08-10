import rawpy
from load_model import settings
from utils import *
# from mlp import MLP  # why there is an import error?
from sklearn.neural_network import MLPClassifier


def get_rggb(path):
    raw = rawpy.imread(path)
    rggb = pack_raw(raw)
    return rggb


def preprocess(rggb):
    rggb = rggb.transpose(2, 0, 1)
    rggb = norm(rggb)
    return rggb


def train(rgbs):
    train, gt = rgbs
    train = preprocess(train)
    gt = preprocess(gt)
    model, optimizer, lr_scheduler = settings()
