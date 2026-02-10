# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import argparse
import logging
from typing import Optional, List, Dict, Any, Callable

import datasets
import pytorch_lightning as pl
import torch
from datasets import load_dataset, DatasetDict, Image
from torch.utils.data import DataLoader
from torchvision import transforms  # torchvision for image transforms
from transformers import AutoImageProcessor  # Using AutoImageProcessor

from path_constants import HUGGINGFACE_CACHE

logger = logging.getLogger(__name__)


class ImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Image Classification datasets.
    Handles loading, preprocessing with augmentations, and batching.
    """

    # standard column names
    image_column_name: str = "image"
    label_column_name: str = "label"

    dataset_label_field_map = {
        "tanganke/dtd": "label",
        "tanganke/sun397": "label",
        "tanganke/eurosat": "label",
        "tanganke/resisc45": "label",
        "tanganke/gtsrb": "label",
        "ufldl-stanford/svhn": "label",
    }

    dataset_label_type_map = {
        "tanganke/dtd": "cat",
        "tanganke/sun397": "cat",
        "tanganke/eurosat": "cat",
        "tanganke/resisc45": "cat",
        "tanganke/gtsrb": "cat",
        "ufldl-stanford/svhn": "cat",
    }

    def __init__(
        self,
        model_name_or_path: str,  # Needed for image processor config (size, mean, std)
        base_args: argparse.Namespace,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        data_dir: Optional[str] = None,  # For datasets needing local files
        train_split_name: str = "train",
        val_split_name: str = "validation",
        test_split_name: str = "test",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        train_dataloader_num_workers: int = 4,
        eval_dataloader_num_workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_args"])
        self.base_args = base_args

        # Load Image Processor to get size, mean, std for transforms
        logger.info(f"Loading image processor for {model_name_or_path}")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        self.img_size = self.image_processor.size["height"]  # Assuming square
        self.img_mean = self.image_processor.image_mean
        self.img_std = self.image_processor.image_std

        logger.info(f"Inferred Image Size: {self.img_size}")
        logger.info(f"Inferred Image Mean: {self.img_mean}")
        logger.info(f"Inferred Image Std: {self.img_std}")

        # Data loading parameters
        self.data_dir = data_dir

        # Will be set in setup
        self.num_labels = None
        self.label_names = None
        self.dataset = None
        self.train_transforms = None
        self.eval_transforms = None

    def prepare_data(self):
        """Downloads datasets."""
        logger.info(
            f"Preparing data: downloading dataset '{self.hparams.dataset_name}' ({self.hparams.dataset_config_name or 'default'})"
        )
        # Only download, processing happens in setup
        load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config_name,
            cache_dir=HUGGINGFACE_CACHE,
            data_dir=self.data_dir,
            # trust_remote_code=True, if dataset requires it
        )
        logger.info("Data preparation finished.")

    def split_validation_or_test(
        self, full_dataset: DatasetDict, split_ratio: float = 0.5, split_seed: int = 7
    ) -> DatasetDict:
        """
        Checks for validation and test splits. If exactly one exists,
        splits it reproducibly to create the other. Minimal version.

        Args:
            full_dataset: The loaded DatasetDict.

        Returns:
            The potentially modified DatasetDict with train/val/test splits.
        """
        val_key = self.hparams.val_split_name
        test_key = self.hparams.test_split_name
        has_val = val_key in full_dataset
        has_test = test_key in full_dataset

        # Only act if exactly one split exists (XOR)
        if has_val ^ has_test:
            existing_key = val_key if has_val else test_key
            missing_key = test_key if has_val else val_key
            # Using val_test_split_ratio for the size of the test set portion
            # Using seed from base_args if it exists, otherwise from hparams
            split_seed = getattr(self.base_args, "seed", split_seed)

            logger.warning(
                f"Only '{existing_key}' found. Splitting to create '{missing_key}' "
                f"(test ratio={split_ratio}, seed={split_seed})."
            )

            # Perform the split. test_size creates the 'test' key in the output dict.
            split_output = full_dataset[existing_key].train_test_split(
                test_size=split_ratio, seed=split_seed, shuffle=True
            )

            # Create the new dataset dictionary, preserving other splits
            new_dataset_dict = DatasetDict(
                {k: v for k, v in full_dataset.items() if k != existing_key}
            )

            # Assign the results: 'train' key from split becomes validation, 'test' key becomes test
            new_dataset_dict[val_key] = split_output["train"]
            new_dataset_dict[test_key] = split_output["test"]

            logger.info(
                f"Created splits: '{val_key}' ({len(new_dataset_dict[val_key])}), '{test_key}' ({len(new_dataset_dict[test_key])})."
            )
            return new_dataset_dict
        elif (
            not has_val
            and not has_test
            and self.dataset_label_type_map.get(self.hparams.dataset_name) == "cat"
        ):
            # split the dataset into train, validation, and test
            logger.warning(
                f"Neither '{val_key}' nor '{test_key}' found. Splitting 'train' into train, val, test."
            )
            # Using the train split to create both validation and test splits
            split_output = full_dataset[self.hparams.train_split_name].train_test_split(
                test_size=0.2, seed=split_seed, shuffle=True
            )
            split_output_2 = split_output["test"].train_test_split(
                test_size=0.5, seed=split_seed, shuffle=True
            )

            # Create the new dataset dictionary, preserving other splits
            new_dataset_dict = DatasetDict(
                {
                    self.hparams.train_split_name: split_output["train"],
                    self.hparams.val_split_name: split_output_2["train"],
                    self.hparams.test_split_name: split_output_2["test"],
                }
            )
            return new_dataset_dict
        else:
            # If both exist or neither exists, return the original dataset unchanged
            return full_dataset

    def cast_column_to_image(self, dataset_dict_in: DatasetDict, path_column: str):
        updated_splits = {}
        for split_name, ds in dataset_dict_in.items():
            logger.info(
                f"Casting column '{path_column}' to Image(decode=True) for split '{split_name}'..."
            )
            updated_splits[split_name] = ds.cast_column(path_column, Image(decode=True))
            updated_splits[split_name] = updated_splits[split_name].rename_column(
                path_column, self.image_column_name
            )
        dataset_dict = DatasetDict(updated_splits)
        return dataset_dict

    def setup(self, stage: Optional[str] = None):
        """Loads datasets, creates class mappings, defines transforms."""
        logger.info(f"Setting up data for stage: {stage}")

        # 1. Load Dataset
        logger.info(
            f"Loading dataset: {self.hparams.dataset_name} ({self.hparams.dataset_config_name or 'default'})"
        )
        # Use trust_remote_code=True if needed
        full_dataset = load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config_name,
            cache_dir=HUGGINGFACE_CACHE,
            data_dir=self.data_dir,
            trust_remote_code=True,
        )

        full_dataset = self.split_validation_or_test(full_dataset)

        # 2. Get Class Labels and Number
        # Assumes 'train' split exists and has label features
        train_split_key = self.hparams.train_split_name
        if train_split_key not in full_dataset:
            raise ValueError(
                f"Train split '{train_split_key}' not found in dataset '{self.hparams.dataset_name}'."
            )

        try:
            labels_feature = full_dataset[train_split_key].features[
                self.dataset_label_field_map[self.hparams.dataset_name]
            ]
            if isinstance(labels_feature, datasets.ClassLabel):
                self.num_labels = labels_feature.num_classes
                self.label_names = labels_feature.names
                logger.info(f"Found {self.num_labels} classes: {self.label_names}")
            #  self.label_to_int = {name: i for i, name in enumerate(self.label_names)}
            else:  # Handle cases like multi-label or simple integer labels without names
                # Infer from unique values (can be slow for large datasets)
                logger.warning(
                    f"Label column '{self.dataset_label_field_map[self.hparams.dataset_name]}' is not ClassLabel type. Attempting to infer num_labels."
                )
                # Check a subset for unique labels for efficiency
                unique_labels = list(
                    set(
                        full_dataset[train_split_key][
                            self.dataset_label_field_map[self.hparams.dataset_name]
                        ]
                    )
                )
                label_feature = datasets.ClassLabel(names=unique_labels)
                full_dataset = full_dataset.cast_column(
                    self.dataset_label_field_map[self.hparams.dataset_name],
                    label_feature,
                )
                # Step 4: Extract metadata
                self.num_labels = label_feature.num_classes
                self.label_names = label_feature.names
                logger.info(
                    f"Converted to ClassLabel with {self.num_labels} classes: {self.label_names}"
                )
        except Exception as e:
            raise ValueError(
                f"Could not determine number of labels from dataset '{self.hparams.dataset_name}'. Error: {e}"
            )

        # 3. Define Transformations
        # Training transforms with augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        )

        # Eval/Test transforms (no augmentation)
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize(self.img_size),  # Simple resize for eval
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        )

        # 4. Select Splits and Set Transforms
        # Use .set_transform for on-the-fly processing (more memory efficient)
        self.dataset = DatasetDict()
        split_mapping = {
            "train": (self.hparams.train_split_name, self.train_transforms),
            "validation": (self.hparams.val_split_name, self.eval_transforms),
            "test": (self.hparams.test_split_name, self.eval_transforms),
        }

        for stage_name, (split_key, transform) in split_mapping.items():
            if split_key in full_dataset:
                logger.info(f"Setting transform for {stage_name} split ('{split_key}')")
                current_ds = full_dataset[split_key]
                # Ensure image column is decoded before transform
                if isinstance(current_ds.features.get(self.image_column_name), Image):
                    current_ds = current_ds.cast_column(
                        self.image_column_name, Image(decode=True)
                    )

                current_ds.set_transform(
                    lambda batch: self._transform_batch(batch, transform)
                )
                self.dataset[stage_name] = current_ds
            else:
                logger.warning(
                    f"{stage_name.capitalize()} split '{split_key}' not found in dataset."
                )

        if not self.dataset:
            raise ValueError("No valid dataset splits found or processed.")
        logger.info(f"Processed dataset splits: {list(self.dataset.keys())}")

    def _transform_batch(
        self, batch: Dict[str, Any], transform: Callable
    ) -> Dict[str, Any]:
        """
        Applies the transform to the image column and returns ONLY the processed
        pixel values (tensors) and labels, ready for default_collate.
        """
        output = {}

        # Process images: Convert to RGB, apply transforms (ToTensor, Normalize, etc.)
        # Assumes batch[img_col] contains a list of PIL Images
        output["pixel_values"] = [
            transform(img.convert("RGB")) for img in batch[self.image_column_name]
        ]

        # Getting labels directly from the input batch using the correct column name
        output["labels"] = batch[
            self.dataset_label_field_map[self.hparams.dataset_name]
        ]

        # Return dictionary containing only the required processed tensors/labels
        return output

    # --- DataLoader Methods ---
    def train_dataloader(self):
        if "train" not in self.dataset:
            return None
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            collate_fn=torch.utils.data.default_collate,
            num_workers=self.hparams.train_dataloader_num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if "validation" not in self.dataset:
            return None
        # Can return multiple validation loaders if needed (e.g., for different domains)
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            collate_fn=torch.utils.data.default_collate,
            num_workers=self.hparams.eval_dataloader_num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if "test" not in self.dataset:
            return None
        return DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            collate_fn=torch.utils.data.default_collate,
            num_workers=self.hparams.eval_dataloader_num_workers,
            pin_memory=True,
        )
