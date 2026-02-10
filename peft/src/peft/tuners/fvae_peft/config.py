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

from ..lora import LoraConfig
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.utils import PeftType


@dataclass
class FVAEPEFTConfig(LoraConfig):
    """fvae peft config"""

    # input_dim=input_hidden_size,
    latent_dim: int = field(default=128, metadata={"help": "latent dim"})
    num_of_z_sampled: int = field(default=1, metadata={"help": "number of z sampled"})
    latent_fusion: str = field(default="concat", metadata={"help": "latent fusion"})
    #
    encoder_nn_type: str = field(default="mlp", metadata={"help": "encoder nn type"})
    enc_num_of_layer: int = field(default=3, metadata={"help": "encoder number of layer"})
    enc_hidden_layer: Union[list[int], int] = field(
        default=1024, metadata={"help": "enc hidden layers, scalar or list for each layer"}
    )
    enc_dropout: float = field(default=0.1, metadata={"help": "encoder dropout"})
    encoder_use_common_hidden_layer: bool = field(default=True, metadata={"help": "encoder use common hidden layer"})
    enc_l2_reg: float = field(default=0, metadata={"help": "encoder l2 reg"})
    #
    decoder_nn_type: str = field(default="mlp", metadata={"help": "decoder nn type"})
    dec_num_of_layer: int = field(default=3, metadata={"help": "decoder number of layer"})
    dec_hidden_layer: Union[list[int], int] = field(
        default=1024, metadata={"help": "dec hidden layers, scalar or list for each layer"}
    )
    dec_dropout: float = field(default=0.1, metadata={"help": "decoder dropout"})
    dec_l2_reg: float = field(default=0, metadata={"help": "decoder l2 reg"})
    scalar_decoder_std: bool = field(default=True, metadata={"help": "scalar decoder std"})
    # Position of z2 in latent space
    z2_latent_mean: float = field(default=1.5, metadata={"help": "z2 latent mean"})
    z2_latent_std: float = field(default=1, metadata={"help": "z2 latent std"})
    # Regularizer
    z1z2_orthogonal_reg: float = field(default=0, metadata={"help": "z1z2 orthogonal reg"})

    # test functionality
    force_classical_vae: bool = field(default=False, metadata={"help": "force classical vae"})
    force_classical_ae: bool = field(default=False, metadata={"help": "force classical ae"})
    force_test_vae: bool = field(default=False, metadata={"help": "force test vae"})
    force_minhyp: bool = field(default=False, metadata={"help": "force minhyp"})

    # to declare in model
    # z1_latent_mean = 0 * torch.ones(1, latent_dim)
    # z1_latent_std = 1 * torch.ones(1, latent_dim)

    # z2_latent_mean = z2_latent_mean * torch.ones(1, latent_dim)
    # z2_latent_std = z2_latent_std * torch.ones(1, latent_dim)

    # weights in new elbo
    lambda_downstream: float = field(default=1, metadata={"help": "lambda downstream, most likely CE"})
    lambda_reconstruction: float = field(default=1, metadata={"help": "ELBO FVAE: lambda reconstruction"})
    lambda_z2_l2: float = field(default=1, metadata={"help": "ELBO FVAE: lambda z2 l2"})
    lambda_z1_l2: float = field(default=1, metadata={"help": "ELBO FVAE: lambda z1 l2"})
    lambda_kl_z1: float = field(default=1, metadata={"help": "ELBO FVAE: lambda kl z1"})

    def __post_init__(self):
        self.peft_type = PeftType.FVAEPEFT
