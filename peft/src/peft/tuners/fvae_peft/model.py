# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import warnings
import torch
from transformers.pytorch_utils import Conv1D

from peft.utils import (
    TRANSFORMERS_MODELS_TO_FVAEPEFT_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)

from peft.utils.integrations import gather_params_ctx

from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer

from .fvae_peft_mlp_layer import FVAEPEFTLoraLayer, FVAEPEFTLoraLayerWrapped


class FVAEPEFTLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

        self.overall_config = config

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise ImportError(
                "To use AdaLora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        # If it is not an AdaLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, FVAEPEFTLoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear) or isinstance(
            target_base_layer, torch.nn.modules.linear.Linear
        ):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target}, type: {type(target)} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = FVAEPEFTLoraLayerWrapped(target, adapter_name, lora_config, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_FVAEPEFT_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_FVAEPEFT_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        if not self.training:
            return outputs
        # outputs.loss = 100 * outputs.loss
        outputs.loss = self.overall_config.lambda_downstream * outputs.loss
        fvae_losses = 0
        total_fvae_modules = 0

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
            # Calculate the orthogonal regularization
            # orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight

            # if orth_reg_weight <= 0:
            #     raise ValueError("orth_reg_weight should be greater than 0. ")

            # regu_loss = 0
            # num_param = 0
            # for n, p in self.model.named_parameters():
            #     if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
            #         if p.shape == torch.Size([0]):
            #             with gather_params_ctx(p, fwd_module=self):
            #                 para_cov = p @ p.T if "lora_A" in n else p.T @ p
            #         else:
            #             para_cov = p @ p.T if "lora_A" in n else p.T @ p
            #         I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))  # noqa: E741
            #         I.requires_grad = False
            #         num_param += 1
            #         regu_loss += torch.norm(para_cov - I, p="fro")
            # if num_param > 0:
            #     regu_loss = regu_loss / num_param
            # else:
            #     regu_loss = 0
            # outputs.loss += orth_reg_weight * regu_loss
            # for n, p in self.model.named_parameters():
            #     if "fvae_peft_current_elbo" in n:
            #         outputs.loss += p
            for l, k in self.model.named_modules():
                if isinstance(k, FVAEPEFTLoraLayerWrapped):
                    # outputs.loss += 0.01 * k.fvae_peft_current_elbo
                    fvae_losses += k.fvae_peft_current_elbo
                    # print(f"after fvae losses: {fvae_losses.requires_grad}")
                    total_fvae_modules += 1
        if total_fvae_modules > 0:
            outputs.loss += fvae_losses / total_fvae_modules
        # print(fvae_losses)
        # print(fvae_losses.requires_grad)
        # import pdb

        # pdb.set_trace()
        return outputs
