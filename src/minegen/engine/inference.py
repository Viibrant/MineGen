import logging
from typing import Any

import torch
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy

from ..config.models import Config


def inference(cfg: Config, model: torch.nn.Module, val_loader: Any) -> None:
    """Inference function for the model"""

    device = cfg.model.device

    logger = logging.getLogger("schematic.inference")
    logger.info("Start inference")

    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy()}, device=device
    )

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        logger.info(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f}".format(
                engine.state.epoch, avg_accuracy
            )
        )

    evaluator.run(val_loader)