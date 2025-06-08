from .schem_model import SchemNet
from .conv_vae import VAE


def build_model(cfg):
    model = SchemNet(cfg.MODEL.NUM_CLASSES)
    return model
