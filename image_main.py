# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import custom modules
from image_datamodule import ImageDataModule
from image_model import ImagePeftModel

# Import PEFT configs
from peft import LoraConfig, FVAEPEFTConfig

from path_constants import HUGGINGFACE_CACHE, LARGE_MODELS_PATH


def main(args: argparse.Namespace):
    # Setting up seed and necessary directories
    pl.seed_everything(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = (
        args.exp_dir
        / f"training_log_{args.dataset_name.split('/')[-1]}_{args.train_split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Run: {args.exp_dir}")
    logger.info(f"All arguments: {vars(args)}")
    args.meta_logger = logger

    # --- PEFT Configurations ---
    # Defaults for ViT models
    default_vit_targets = [
        "query",
        "value",
    ]  # Common targets in ViT attention/MLP
    default_convnext_targets = ["Conv2d"]  # Targeting convolutional layers

    if "peft" in args.fine_tuning_method or args.fine_tuning_method in [
        "pissa",
        "dora",
        "rslora",
        "olora",
    ]:
        target_modules = args.lora_target_modules
        if not target_modules:
            logger.warning(
                "No target modules specified for LoRA. Attempting defaults..."
            )
            if "vit" in str(args.model_path).lower():
                target_modules = default_vit_targets
                logger.info(f"Using default ViT LoRA targets: {target_modules}")
            elif "convnext" in str(args.model_path).lower():
                target_modules = default_convnext_targets
                logger.info(f"Using default ConvNeXt LoRA targets: {target_modules}")
            # Add other model type checks if needed (e.g., clip, resnet)
            elif "clip" in str(args.model_path).lower():  # CLIP uses ViT architecture
                target_modules = default_vit_targets
                logger.info(
                    f"Using default ViT LoRA targets for CLIP model: {target_modules}"
                )
            else:
                logger.error(
                    "Cannot determine default LoRA targets. Specify --lora_target_modules."
                )
                raise ValueError("Missing LoRA target modules")

    if args.fine_tuning_method == "peft":
        args.lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["classifier"],  # Important: Save the classifier head!
        )
        logger.info(f"LoRA Config: {args.lora_config}")
    elif args.fine_tuning_method == "pissa":
        logger.info("Using PISSA variant of LoRA.")
        args.lora_config = LoraConfig(
            r=args.lora_r,  # Rank for the low-rank matrix
            lora_alpha=args.lora_alpha,  # Scaling factor
            target_modules=target_modules,
            init_lora_weights="pissa",  # Initialize LoRA weights with PISSA
            lora_dropout=0.0,
            bias="none",  # No bias in LoRA layers
            modules_to_save=["classifier"],
        )
    elif args.fine_tuning_method == "dora":
        logger.info("Using DORA variant of LoRA.")
        args.lora_config = LoraConfig(
            r=args.lora_r,  # Rank for the low-rank matrix
            lora_alpha=args.lora_alpha,  # Scaling factor
            target_modules=target_modules,
            use_dora=True,  # Use DORA
            lora_dropout=args.lora_dropout,
            bias="none",  # No bias in LoRA layers
            modules_to_save=["classifier"],
        )
    elif args.fine_tuning_method == "rslora":
        logger.info("Using RSLORA variant of LoRA.")
        args.lora_config = LoraConfig(
            r=args.lora_r,  # Rank for the low-rank matrix
            lora_alpha=args.lora_alpha,  # Scaling factor
            target_modules=target_modules,
            use_rslora=True,  # Use RSLORA
            lora_dropout=args.lora_dropout,
            bias="none",  # No bias in LoRA layers
            modules_to_save=["classifier"],
        )
    elif args.fine_tuning_method == "olora":
        args.lora_config = LoraConfig(
            r=args.lora_r,  # Rank for the low-rank matrix
            lora_alpha=args.lora_alpha,  # Scaling factor
            init_lora_weights="olora",  # Initialize LoRA weights with PISSA
            lora_dropout=args.lora_dropout,
            bias="none",  # No bias in LoRA layers
            modules_to_save=["classifier"],
            target_modules=target_modules,
        )
    else:
        args.lora_config = None

    if args.fine_tuning_method == "fvae_peft":
        args.fvae_peft_config = FVAEPEFTConfig(
            # Parameters for fvae-lora (fvae-peft)
            peft_type="FVAE_PEFT",  # Accepted by FVAEPEFTConfig
            latent_dim=args.lora_r,  # Using lora_r as latent dim
            latent_fusion="concat",
            enc_num_of_layer=1,
            enc_hidden_layer=args.lora_r,  # 128
            enc_dropout=0.1,
            encoder_use_common_hidden_layer=True,
            dec_num_of_layer=3,
            dec_hidden_layer=128,
            z2_latent_mean=1.5,
            z2_latent_std=1,
            z1z2_orthogonal_reg=0,
            lambda_downstream=args.fvae_lambda_downstream,
            lambda_reconstruction=args.fvae_lambda_reconstruction,
            lambda_z2_l2=args.fvae_lambda_z2_l2,
            lambda_z1_l2=args.fvae_lambda_z1_l2,
            lambda_kl_z1=args.fvae_lambda_kl_z1,
            target_modules=target_modules,
            modules_to_save=["classifier"],  # Also save classifier head here
        )
        logger.info(f"FVAE-PEFT Config: {vars(args.fvae_peft_config)}")
    else:
        args.fvae_peft_config = None

    # --- DataModule ---
    dm = ImageDataModule(
        model_name_or_path=args.model_path,  # model path required for processor config
        base_args=args,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        data_dir=args.data_dir,
        train_split_name=args.train_split_name,
        val_split_name=args.val_split_name,
        test_split_name=args.test_split_name,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_dataloader_num_workers=args.num_workers_dataloader_train,
        eval_dataloader_num_workers=args.num_workers_dataloader_eval,
    )

    # Setting data module before model initialization to get num_labels
    dm.setup(stage="fit")  # Or stage=None if testing immediately
    if dm.num_labels is None:
        raise ValueError("Number of labels could not be determined by DataModule.")

    # --- Model ---
    model = ImagePeftModel(
        model_name_or_path=args.model_path,
        num_labels=dm.num_labels,  # Get num_labels from datamodule
        base_args=args,
        learning_rate=args.lr,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    # loging model to inspect
    logger.info(model)

    # --- Callbacks ---
    callbacks = []
    # Monitor validation accuracy (higher is better)
    checkpoint_monitor = "val_acc"  # Changed from val_loss
    checkpoint_filename = "{epoch}-{val_acc:.4f}"  # filename for saving

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.exp_dir / "checkpoints",
        filename=checkpoint_filename,
        monitor=checkpoint_monitor,
        save_top_k=args.save_top_k,
        mode="max",  # Maximize accuracy
        save_weights_only=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor=checkpoint_monitor,
            patience=args.early_stopping_patience,
            verbose=True,
            mode="max",  # Maximize accuracy
        )
        callbacks.append(early_stop_callback)

    # --- Loggers ---
    args.loggers = []
    args.meta_logger = logging.getLogger(__name__)

    if args.wandb:
        model_type_safe = Path(args.model_path).name  # Safer way to get model name
        wandb_name = f"{args.dataset_name}_{args.fine_tuning_method}_{model_type_safe}_seed{args.seed}"
        if args.dataset_config_name:
            wandb_name = f"{args.dataset_name}-{args.dataset_config_name}_{args.fine_tuning_method}_{model_type_safe}_seed{args.seed}"

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=wandb_name,
            group=f"{args.dataset_name}-{args.dataset_config_name or 'all'}",
            save_dir=args.exp_dir / "wandb_logs",
            log_model=False,  # don't log model to wandb with PL checkpoints
        )
        hyperparams_to_log = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        # Add num_labels from datamodule to logged hparams
        hyperparams_to_log["num_labels"] = dm.num_labels
        wandb_logger.log_hyperparams(hyperparams_to_log)
        args.loggers.append(wandb_logger)
        logger.info(
            f"WandB Logger initialized: project='{args.wandb_project}', name='{wandb_name}'"
        )

    # TensorBoard Logger setup
    if args.tensorboard_use:
        model_type = str(args.model_path).split("/")[-1]
        tb_version = f"{args.fine_tuning_method}_{model_type}_{args.seed}"
        if args.dataset_config_name:
            tb_version = f"{args.dataset_config_name}_{tb_version}"

        tensorboard_logger = TensorBoardLogger(
            save_dir=args.tensorboard_logdir,
            name=args.dataset_name,
            version=tb_version,
            default_hp_metric=False,  # Avoids potential issues with hp logging
        )
        args.loggers.append(tensorboard_logger)
        logger.info(
            f"TensorBoard Logger initialized: save_dir='{args.tensorboard_logdir}', name='{args.dataset_name}', version='{tb_version}'"
        )

    if not args.loggers:
        logger.warning(
            "No logger specified (WandB or TensorBoard). Training progress will only be in console/logfile."
        )

    # --- Trainer ---
    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.num_gpus if torch.cuda.is_available() else 1,
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
        logger=args.loggers if args.loggers else True,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    # --- Training ---
    logger.info("Starting training...")
    trainer.fit(model, datamodule=dm)

    # --- Testing ---
    if args.run_test_after_train:
        logger.info("Starting testing using the best checkpoint...")
        # Note that Trainer automatically loads the best checkpoint based on monitor
        trainer.test(model, datamodule=dm, ckpt_path="best")

    logger.info(f"Run finished. Results saved in: {args.exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune vit models with PyTorch Lightning and PEFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment Setup ---
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Base directory for experiment outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--disable_hf_cache", action="store_true", help="Disable HF dataset caching."
    )

    # --- Model ---
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Path or name of pre-trained image model.",
    )

    # --- Dataset ---
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of HF dataset (e.g., 'tanganke/dtd', 'tanganke/eurosat').",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Config name for dataset (e.g., subset for ufldl-stanford/svhn).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Local directory for datasets if already downloaded.",
    )
    parser.add_argument(
        "--train_split_name", type=str, default="train", help="Training split name."
    )
    parser.add_argument(
        "--val_split_name",
        type=str,
        default="validation",
        help="Validation split name (e.g., 'validation', 'test').",
    )
    parser.add_argument(
        "--test_split_name", type=str, default="test", help="Test split name."
    )
    # Image/Label column names are now class attributes in datamodule

    # --- Training ---
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Evaluation batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Num linear warmup steps (overridden by ratio).",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of total steps for warmup.",
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision.",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Validate every N epochs.",
    )

    # --- Dataloading ---
    parser.add_argument(
        "--num_workers_dataloader_train",
        type=int,
        default=14,
        help="Train dataloader workers.",
    )
    parser.add_argument(
        "--num_workers_dataloader_eval",
        type=int,
        default=14,
        help="Eval dataloader workers.",
    )
    # preprocessing_num_workers not strictly needed if using set_transform

    # --- Fine-tuning Method ---
    parser.add_argument(
        "--fine_tuning_method",
        type=str,
        choices=[
            "full_ft",
            "peft",
            "fvae_peft",
            "custom",
            "pissa",
            "dora",
            "rslora",
            "olora",
        ],
        default="full_ft",
        help="Fine-tuning approach.",
    )

    # --- PEFT LoRA Specific ---
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank 'r'.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Module names for LoRA. Defaults based on model type.",
    )

    # --- FVAE-LoRA (FVAE-PEFT) Specific ---
    parser.add_argument(
        "--fvae_lambda_downstream",
        type=float,
        default=1000,
        help="Lambda for downstream task.",
    )
    parser.add_argument(
        "--fvae_lambda_reconstruction",
        type=float,
        default=1,
        help="Lambda for reconstruction.",
    )
    parser.add_argument(
        "--fvae_lambda_z2_l2",
        type=float,
        default=1,
        help="Lambda for z2 L2 regularization.",
    )
    parser.add_argument(
        "--fvae_lambda_z1_l2",
        type=float,
        default=1,
        help="Lambda for z1 L2 regularization.",
    )
    parser.add_argument(
        "--fvae_lambda_kl_z1",
        type=float,
        default=1,
        help="Lambda for KL divergence of z1.",
    )

    # --- Logging & Checkpointing ---
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument(
        "--wandb_project", type=str, default="", help="WandB project name."
    )
    parser.add_argument(
        "--tensorboard_use", action="store_true", help="Enable TensorBoard logging."
    )
    parser.add_argument(
        "--tensorboard_logdir",
        type=Path,
        default="./tensorboard_logs/",
        help="TensorBoard log dir.",
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50, help="Log metrics every N steps."
    )
    parser.add_argument(
        "--save_top_k", type=int, default=1, help="Save top K checkpoints."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping (monitors val_acc).",
    )
    parser.add_argument(
        "--run_test_after_train",
        action="store_true",
        help="Run test set evaluation after training.",
    )

    args = parser.parse_args()

    args.model_path = LARGE_MODELS_PATH / args.model_path
    # Append seed to experiment directory
    args.exp_dir = args.exp_dir  # / f"seed_{args.seed}"
    if args.dataset_config_name is None and args.dataset_name == "ufldl-stanford/svhn":
        args.dataset_config_name = "cropped_digits"

    main(args)
