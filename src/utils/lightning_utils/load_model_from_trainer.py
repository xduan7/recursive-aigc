from copy import deepcopy
from typing import Literal, Optional

import pytorch_lightning as pl
import torch


def _get_checkpoint_callback_model_path_from_trainer(
    trainer: pl.Trainer,
    monitor: Optional[str],
    callback_type: Literal["best", "last"],
) -> Optional[pl.callbacks.ModelCheckpoint]:
    """Get the (best/last) model path from a PyTorch Lightning Trainer."""
    _path_attr = (
        "best_model_path" if callback_type == "best" else "last_model_path"
    )
    if monitor is None:
        return getattr(trainer.checkpoint_callback, _path_attr, None)
    model_path = None
    for __cb in trainer.checkpoint_callbacks:
        if not isinstance(__cb, pl.callbacks.ModelCheckpoint):
            continue
        if __cb.monitor == monitor:
            model_path = getattr(__cb, _path_attr, None)
            break
    return model_path


def load_best_model_from_trainer(
    trainer: pl.Trainer,
    monitor: Optional[str] = None,
) -> pl.LightningModule:
    """Load the best model from a PyTorch Lightning Trainer.

    Args:
        trainer: A PyTorch Lightning Trainer with checkpoint callback after
            training for at least one epoch.
        monitor: The metric to monitor. If None, the default metric of the
            Trainer is used.

    Returns:
        A PyTorch Lightning Module loaded from the best model path.

    """
    _lightning_module = deepcopy(trainer.lightning_module)
    _best_model_path = _get_checkpoint_callback_model_path_from_trainer(
        trainer=trainer,
        monitor=monitor,
        callback_type="best",
    )
    if _best_model_path is None:
        raise ValueError(
            f"Could not find a checkpoint callback with monitor {monitor}"
        )
    _best_model_state_dict = torch.load(_best_model_path)["state_dict"]
    _lightning_module.load_state_dict(_best_model_state_dict)
    return _lightning_module


def load_last_model_from_trainer(
    trainer: pl.Trainer,
    monitor: Optional[str] = None,
) -> pl.LightningModule:
    """Load the last model from a PyTorch Lightning Trainer.

    Args:
        trainer: A PyTorch Lightning Trainer with checkpoint callback after
            training for at least one epoch.
        monitor: The metric to monitor. If None, the default metric of the
            Trainer is used.

    Returns:
        A PyTorch Lightning Module loaded from the last model path.

    """
    _lightning_module = deepcopy(trainer.lightning_module)
    _last_model_path = _get_checkpoint_callback_model_path_from_trainer(
        trainer=trainer,
        monitor=monitor,
        callback_type="last",
    )
    if _last_model_path is None:
        raise ValueError(
            "Could not find a checkpoint callback with valid last model path"
        )
    _last_model_state_dict = torch.load(_last_model_path)["state_dict"]
    _lightning_module.load_state_dict(_last_model_state_dict)
    return _lightning_module
