import logging

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, Timer


def train(cfg, model, train_loader, val_loader, optimiser, scheduler, loss_fn):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("schematic.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimiser, loss_fn, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )

    checkpointer = ModelCheckpoint(
        output_dir,
        "schematic",
        save_interval=checkpoint_period,
        n_saved=10,
        require_empty=False,
    )
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        logger.info(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_loss
            )
        )

    if val_loader is not None:

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_loss = metrics["loss"]
            logger.info(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_loss
                )
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_time(engine):
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s]".format(
                engine.state.epoch, timer.value()
            )
        )
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
