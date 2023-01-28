from .schem_model import SchemNet


def build_model(cfg):
    model = SchemNet(cfg.MODEL.NUM_CLASSES)
    return model
