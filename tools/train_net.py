"""Code to train the network"""
import torch.nn.functional as F

from data import make_data_loader
from engine.trainer import train
from modelling import build_model
from solver import make_optimiser


def train_net(cfg):
    model = build_model(cfg)
    optimiser = make_optimiser(cfg, model)
    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    train(cfg, model, train_loader, val_loader, optimiser, None, F.cross_entropy)
