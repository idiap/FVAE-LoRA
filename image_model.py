# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import argparse
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import evaluate  # Can use for accuracy, but torchmetrics is often easier
import torchmetrics  # Use torchmetrics for easy logging

from peft import get_peft_model, LoraConfig, PeftModel

# from fvae_peft_config import FVAEPEFTConfig

from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class ImagePeftModel(pl.LightningModule):
    """
    PyTorch Lightning Module for fine-tuning Image Classification models
    with optional PEFT methods (LoRA, FVAE-PEFT) or full fine-tuning.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,  # Required for classification head
        base_args: argparse.Namespace,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        warmup_ratio: Optional[float] = 0.1,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base_args = base_args

        self.num_labels = num_labels

        logger.info(f"Loading config for {model_name_or_path}")
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
        )
        logger.info(
            f"Loading model {model_name_or_path} for Image Classification with {self.num_labels} labels."
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            ignore_mismatched_sizes=True,
        )

        # --- PEFT / Fine-tuning Setup ---
        self.hparams.base_args.meta_logger = logger

        if self.hparams.base_args.fine_tuning_method == "full_ft":
            self.hparams.base_args.meta_logger.info("Fine-tuning the full image model.")
            self.model = model
        elif (
            self.hparams.base_args.fine_tuning_method == "peft"
            or self.hparams.base_args.fine_tuning_method
            in ["pissa", "dora", "rslora", "olora"]
        ):
            self.hparams.base_args.meta_logger.info(
                "Enabling PEFT (LoRA) fine-tuning for image model."
            )
            if (
                not hasattr(self.hparams.base_args, "lora_config")
                or self.hparams.base_args.lora_config is None
            ):
                raise ValueError("LoRA config missing for PEFT method 'peft'.")
            self.model = get_peft_model(model, self.hparams.base_args.lora_config)
            self.model.print_trainable_parameters()
        elif self.hparams.base_args.fine_tuning_method == "fvae_peft":
            self.hparams.base_args.meta_logger.info(
                "Enabling FVAE-LoRA (FVAE-PEFT) fine-tuning."
            )
            if (
                not hasattr(self.hparams.base_args, "fvae_peft_config")
                or self.hparams.base_args.fvae_peft_config is None
            ):
                raise ValueError("FVAE-PEFT config missing.")
            self.model = get_peft_model(model, self.hparams.base_args.fvae_peft_config)
            self.model.print_trainable_parameters()
        # Add elif for 'custom' if implementing specific logic
        else:  # Handles 'custom' if not implemented and any other invalid value
            if self.hparams.base_args.fine_tuning_method == "custom":
                self.hparams.base_args.meta_logger.warning(
                    "Custom fine-tuning selected but not implemented. Using full model."
                )
                self.model = model
            else:
                raise ValueError(
                    f"Fine-tuning method '{self.hparams.base_args.fine_tuning_method}' not recognized."
                )

        # --- Metrics ---
        # Using torchmetrics for easy integration with PyTorch Lightning
        task = "multilabel" if self.num_labels > 1 and False else "multiclass"
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=self.num_labels)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=self.num_labels)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=self.num_labels)

    def forward(
        self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        """Forward pass"""
        return self.model(pixel_values=pixel_values, labels=labels)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Common logic for training, validation, and test steps."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self(pixel_values=pixel_values, labels=labels)
        # If model doesn't return loss when labels are passed, calculate it here:
        if outputs.loss is None:
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)  # Standard Cross Entropy
        else:
            loss = outputs.loss
            logits = outputs.logits

        return {"loss": loss, "logits": logits, "labels": labels}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        step_output = self._shared_step(batch)
        loss = step_output["loss"]
        logits = step_output["logits"]
        labels = step_output["labels"]

        # Log loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Update and log accuracy
        self.train_acc(logits, labels)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        step_output = self._shared_step(batch)
        loss = step_output["loss"]
        logits = step_output["logits"]
        labels = step_output["labels"]

        # Log loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Update accuracy metric state
        self.val_acc(logits, labels)
        # Log accuracy - Lightning handles compute/reset when metric object is logged
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step"""
        step_output = self._shared_step(batch)
        loss = step_output["loss"]
        logits = step_output["logits"]
        labels = step_output["labels"]

        # Log loss
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Update accuracy metric state
        self.test_acc(logits, labels)
        # Log accuracy
        self.log(
            "test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """Prepare optimizer and scheduler"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]  # Common no_decay params

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        # Scheduler
        if self.trainer is None:
            # Estimate steps if trainer not available (e.g. during init) - needs refinement
            logger.warning(
                "Trainer not available in configure_optimizers. Scheduler steps might be inaccurate."
            )
            num_training_steps = (
                10000 * self.hparams.base_args.num_epochs
            )  # Rough estimate
        else:
            num_training_steps = self.trainer.estimated_stepping_batches

        num_warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_ratio is not None and self.hparams.warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)

        logger.info(
            f"Scheduler: Total estimated steps={num_training_steps}, Warmup steps={num_warmup_steps}"
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler_config]
