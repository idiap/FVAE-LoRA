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
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import PretrainedConfig
from torch.distributions import MultivariateNormal
from peft.tuners.lora import LoraLayer, LoraConfig
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


class FVAEPEFTLoraLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "ranknum")

    def __init__(self, base_layer: nn.Module, fave_peft_config: LoraConfig) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})
        self.fave_peft_config = fave_peft_config

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # adjust r
        r = self.fave_peft_config.latent_dim

        # Actual trainable parameters
        # Right singular vectors
        # self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        self.lora_A[adapter_name] = FVAE(self.fave_peft_config, input_dim=self.in_features)
        # Singular values
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
        # Left singular vectors
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))

        # The current rank
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            # nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class FVAEPEFTLoraLayerWrapped(nn.Module, FVAEPEFTLoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fave_peft_config: LoraConfig,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FVAEPEFTLoraLayer.__init__(self, base_layer, fave_peft_config)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.fave_peft_config = fave_peft_config
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        self.fvae_peft_current_elbo = 0

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
            transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out)
            * self.scaling[adapter]
            / (self.ranknum[adapter] + 1e-5)
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # import pdb

        # pdb.set_trace()
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-5

                x = x.to(lora_B.dtype)

                kl_qp, minus_ce_qp, z1z2_orthogonal_reg_term, z1_mean, z2_mean, z1_sample, z2_sample, z1_l2, z2_l2 = (
                    lora_A(x)
                )

                # print(
                #     f"hooks requires grad: {kl_qp.requires_grad}, {minus_ce_qp.requires_grad}, {z1_sample.requires_grad}, {z1_l2.requires_grad}, {z1_sample.requires_grad}"
                # )
                # other formulae
                # minus_elbo = kl_qp + minus_ce_qp + z1z2_orthogonal_reg_term
                minus_elbo = (
                    self.fave_peft_config.lambda_reconstruction * minus_ce_qp
                    + self.fave_peft_config.lambda_z2_l2 * z2_l2
                    - self.fave_peft_config.lambda_z1_l2 * z1_l2
                    + self.fave_peft_config.lambda_kl_z1 * kl_qp
                )
                self.fvae_peft_current_elbo = minus_elbo
                # result += (z1_sample @ lora_B.T) * scaling / ranknum
                if self.training:
                    result += (z1_sample @ lora_B.T) * scaling / ranknum
                else:
                    result += (z1_mean @ lora_B.T) * scaling / ranknum
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fvaepeft." + rep


class FVAEConfig(PretrainedConfig):
    def __init__(
        self,
        #
        input_dim=1024,
        #
        latent_dim=128,
        num_of_z_sampled=1,
        latent_fusion="concat",
        #
        encoder_nn_type="mlp",
        enc_num_of_layer=3,
        enc_hidden_layer=1024,  # scalar or list for each layer
        enc_dropout=0.1,
        encoder_use_common_hidden_layer=True,  # for mean and logvar
        enc_l2_reg=0,
        #
        decoder_nn_type="mlp",
        dec_num_of_layer=3,
        dec_hidden_layer=1024,  # scalar or list for each layer
        dec_dropout=0.5,
        dec_l2_reg=0,
        scalar_decoder_std=True,
        # Position of z2 in latent space
        z2_latent_mean=0,
        z2_latent_std=1,
        # Regularizer
        z1z2_orthogonal_reg=0,
        # test functionnality
        force_classical_vae=False,
        force_classical_ae=False,
        force_test_vae=False,
        force_minhyp=False,
    ):

        super().__init__()
        #
        self.input_dim = input_dim
        # Parameters
        self.latent_dim = latent_dim
        self.num_of_z_sampled = num_of_z_sampled
        self.latent_fusion = latent_fusion  # concat or average
        # Encoders
        self.encoder_nn_type = encoder_nn_type
        self.enc_num_of_layer = enc_num_of_layer
        self.enc_hidden_layer = enc_hidden_layer
        self.enc_dropout = enc_dropout
        self.encoder_use_common_hidden_layer = encoder_use_common_hidden_layer
        self.enc_l2_reg = enc_l2_reg
        # Decoders
        self.decoder_nn_type = decoder_nn_type
        self.dec_num_of_layer = dec_num_of_layer
        self.dec_hidden_layer = dec_hidden_layer
        self.dec_dropout = dec_dropout
        self.dec_l2_reg = dec_l2_reg
        self.scalar_decoder_std = scalar_decoder_std
        # # Position of z2 in latent space
        self.z1_latent_mean = 0 * torch.ones(1, latent_dim)
        self.z1_latent_std = 1 * torch.ones(1, latent_dim)

        self.z2_latent_mean = z2_latent_mean * torch.ones(1, latent_dim)
        self.z2_latent_std = z2_latent_std * torch.ones(1, latent_dim)

        self.z1z2_orthogonal_reg = z1z2_orthogonal_reg

        self.force_classical_vae = force_classical_vae
        self.force_classical_ae = force_classical_ae
        self.force_test_vae = force_test_vae
        self.force_minhyp = force_minhyp


class FVAE(torch.nn.Module):

    config_class = FVAEConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, input_dim):
        super().__init__()
        # self.num_of_views = 2
        self.config = config
        config.input_dim = input_dim
        self.latent_dim = config.latent_dim
        self.num_of_z_sampled = config.num_of_z_sampled
        self.latent_fusion = config.latent_fusion

        # self.z2_latent_mean = config.z2_latent_mean
        # self.z2_latent_var = config.z2_latent_std**2
        self.z2_latent_mean = config.z2_latent_mean * torch.ones(1, self.latent_dim)
        self.z2_latent_var = (config.z2_latent_std * torch.ones(1, self.latent_dim)) ** 2

        self.force_classical_vae = config.force_classical_vae
        self.force_classical_ae = config.force_classical_ae
        self.force_test_vae = config.force_test_vae
        self.force_minhyp = config.force_minhyp

        # Encoders
        if self.force_minhyp:
            self.views_conditionalGaussianEncoders_list = torch.nn.ModuleList(
                [
                    ConditionalGaussianEncodersMLP(  # to model P(z| X)
                        input_dim=config.input_dim,
                        output_dim=config.latent_dim,
                        hidden_layer_size=config.enc_hidden_layer,
                        num_of_layers=config.enc_num_of_layer,
                        dropout=config.enc_dropout,
                        use_common_hidden_layer=config.encoder_use_common_hidden_layer,
                        l2_reg=config.enc_l2_reg,
                    ),
                    ConditionalGaussianEncodersAttention(  # to model P(z2| z1, X)
                        input_dim=config.input_dim,
                        z1_dim=config.latent_dim,
                        output_dim=config.latent_dim,
                        hidden_layer_size=config.enc_hidden_layer,
                        num_of_layers=config.enc_num_of_layer,
                        dropout=config.enc_dropout,
                        use_common_hidden_layer=config.encoder_use_common_hidden_layer,
                        l2_reg=config.enc_l2_reg,
                    ),
                ]
            )
        else:
            self.views_conditionalGaussianEncoders_list = torch.nn.ModuleList(
                [
                    ConditionalGaussianEncodersMLP(  # to model P(z| X)
                        input_dim=config.input_dim,
                        output_dim=config.latent_dim,
                        hidden_layer_size=config.enc_hidden_layer,
                        num_of_layers=config.enc_num_of_layer,
                        dropout=config.enc_dropout,
                        use_common_hidden_layer=config.encoder_use_common_hidden_layer,
                        l2_reg=config.enc_l2_reg,
                    )
                    for k in range(2)
                ]
            )

        # Decoders
        if self.latent_fusion == "concat":
            self.latent_dim_multiplicator = 2
        elif self.latent_fusion == "average":
            self.latent_dim_multiplicator = 1
        else:
            raise ValueError("Invalid latent_fusion")

        if self.force_classical_vae or self.force_test_vae:
            self.latent_dim_multiplicator = 1
        if self.force_classical_vae:
            self.latent_dim_multiplicator = 1

        self.views_conditionalGaussianDecoders = ConditionalGaussianDecodersMLP(
            input_dim=config.latent_dim * self.latent_dim_multiplicator,
            output_dim=config.input_dim,
            hidden_layer_size=config.dec_hidden_layer,
            num_of_layers=config.dec_num_of_layer,
            covariance_matrix_type="diagonal",
            dropout=config.dec_dropout,
            l2_reg=config.dec_l2_reg,
            scalar_decoder_std=config.scalar_decoder_std,
        )

        self.z1z2_orthogonal_reg = config.z1z2_orthogonal_reg

    def forward(self, inputs_values, mask=None) -> torch.Tensor:
        """input_values.shape = (batch_size, input_dim)"""
        # P(X | z1, z2) = N( A mlp([z1, z2]) + a, B mlp([z1, z2]) + b) OR N( A mlp((z1 + z2)) + a, B mlp((z1 + z2)) + b)
        # P(z1, z2) = P(z1) * P(z2) = N(0, I) * N(z2_mean, z2_std)
        # Q(z1, z2 | X) = Q(z1 | X) * Q(z2 | X) = N( C mlp(X) + c, PHI_1) * N( E mlp(X) + e, PHI_2)

        # temp fix: TODO: proper fix
        if mask is None:
            mask = torch.ones(inputs_values.size()[:2], device=inputs_values.device, dtype=torch.int)

        if self.force_classical_vae:
            return self.forward_classical_vae(inputs_values, mask)

        if self.force_classical_ae:
            return self.forward_classical_ae(inputs_values, mask)

        if self.force_test_vae:
            return self.forward_test(inputs_values, mask)

        if self.force_minhyp:
            return self.forward_minhyp(inputs_values, mask)

        z1_mean, z1_logvar = self.views_conditionalGaussianEncoders_list[0](inputs_values, mask)
        # z1_var = torch.exp(z1_logvar)
        z1_var = torch.exp(torch.clamp(z1_logvar, min=-20, max=20))
        z2_mean, z2_logvar = self.views_conditionalGaussianEncoders_list[1](inputs_values, mask)
        # z2_var = torch.exp(z2_logvar)
        z2_var = torch.exp(torch.clamp(z2_logvar, min=-20, max=20))
        # if self.z1z2_orthogonal:
        #     z2_mean = z2_mean - torch.sum(z2_mean * z1_mean, dim=-1, keepdim=True) * z1_mean / torch.sum(z1_mean ** 2 + 1e-6, dim=-1, keepdim=True)

        # Compute Kulback-Leibler : Q log( Q(z1,z2 | X) / P(z1) P(z2)  )  =  Q(z1) log(  Q(z1| X) / P(z1)  )  + Q(z2) log(  Q(z2| X)  / P(z2) )
        kl_qp_1 = self.kullback_leibler_diaggaussian_standardgaussian(
            z1_mean, z1_var
        )  #  Q(z1) log( Q(z1| X)/ P(z1)  )
        kl_qp_2 = self.kullback_leibler_diaggaussian_diaggaussian(
            z2_mean, z2_var, self.z2_latent_mean.to(inputs_values.device), self.z2_latent_var.to(inputs_values.device)
        )  # Q(z2) log( Q(z2 | X) / P(z2)  )

        # [shashi]: new implementation
        # kl_qp = kl_qp_1 + kl_qp_2
        kl_qp = kl_qp_1

        # Cross entropy loss data
        z1_mean = z1_mean.repeat(self.num_of_z_sampled, 1, 1)  # .repeat(self.num_of_z_sampled,1) for MLP
        z1_var = z1_var.repeat(self.num_of_z_sampled, 1, 1)
        z2_mean = z2_mean.repeat(self.num_of_z_sampled, 1, 1)
        z2_var = z2_var.repeat(self.num_of_z_sampled, 1, 1)

        z1_sample = z1_mean + torch.sqrt(z1_var) * torch.randn_like(z1_mean)
        z2_sample = z2_mean + torch.sqrt(z2_var) * torch.randn_like(z2_mean)

        z1_l2 = torch.mean(z1_sample**2)
        z2_l2 = torch.mean((z2_sample - self.config.z2_latent_mean) ** 2)

        # compute decoders parameters
        if self.latent_fusion == "concat":
            z_concat = torch.cat((z1_sample, z2_sample), dim=-1)
            decoders_output = self.views_conditionalGaussianDecoders(z_concat)
        elif self.latent_fusion == "average":
            z_sample = (z1_sample + z2_sample) / 2
            decoders_output = self.views_conditionalGaussianDecoders(z_sample)
        else:
            raise ValueError("Invalid latent_fusion")
        decoders_values, cov_scale_tril = decoders_output

        ce_qp = self.gaussian_cross_entropy(
            decoders_values, inputs_values.repeat(self.num_of_z_sampled, 1, 1), cov_scale_tril
        )
        # ce_qp = self.gaussian_cross_entropy_manual(
        #     decoders_values, inputs_values.repeat(self.num_of_z_sampled, 1, 1), cov_scale_tril
        # )

        # regularizer
        z1z2_orthogonal_reg_term = self.z1z2_orthogonal_reg * torch.mean(
            (z1_mean.transpose(0, 2) @ z2_mean.transpose(0, 2).transpose(1, 2)) ** 2
        )

        extended_mask = mask.repeat(self.num_of_z_sampled, 1)

        # final loss

        return (
            torch.sum(kl_qp * mask) / torch.sum(mask),
            -torch.sum(ce_qp * extended_mask) / torch.sum(extended_mask),
            z1z2_orthogonal_reg_term,
            z1_mean,
            z2_mean,
            z1_sample,
            z2_sample,
            z1_l2,
            z2_l2,
        )

    def forward_classical_vae(self, inputs_values, mask) -> torch.Tensor:
        """input_values.shape = (batch_size, input_dim)"""
        # compute encoders parameters

        if self.force_classical_ae:
            raise ValueError("Error: classical VAE and classical AE set to True in same time")

        z1_mean, z1_logvar = self.views_conditionalGaussianEncoders_list[0](inputs_values, mask)
        z1_var = torch.exp(z1_logvar)

        # Compute Kulback-Leibler
        kl_qp = self.kullback_leibler_diaggaussian_standardgaussian(z1_mean, z1_var)  #  Q log( P(z1) / Q(z1 | X) )

        # Cross entropy loss data
        z1_mean = z1_mean.repeat(self.num_of_z_sampled, 1, 1)  # .repeat(self.num_of_z_sampled,1) for MLP
        z1_var = z1_var.repeat(self.num_of_z_sampled, 1, 1)

        z1_sample = z1_mean + torch.sqrt(z1_var) * torch.randn_like(z1_mean)

        # compute decoders parameters
        decoders_output = self.views_conditionalGaussianDecoders(z1_sample)
        decoders_values, cov_scale_tril = decoders_output

        ce_qp = self.gaussian_cross_entropy(
            decoders_values, inputs_values.repeat(self.num_of_z_sampled, 1, 1), cov_scale_tril
        )
        # ce_qp = self.gaussian_cross_entropy_manual(decoders_values, inputs_values.repeat(self.num_of_z_sampled,1, 1), cov_scale_tril)

        # regularizer
        extended_mask = mask.repeat(self.num_of_z_sampled, 1)
        return (
            torch.sum(kl_qp * mask) / torch.sum(mask),
            -torch.sum(ce_qp * extended_mask) / torch.sum(extended_mask),
            0,
            z1_mean,
            None,
        )

    def forward_classical_ae(self, input_values, mask) -> torch.Tensor:
        if self.force_classical_vae:
            raise ValueError("Error: classical VAE and classical AE set to True in same time")

        z1_mean, _ = self.views_conditionalGaussianEncoders_list[0](inputs_values, mask)
        decoders_values, _ = self.views_conditionalGaussianDecoders(z1_mean)
        return torch.norm((decoders_values - input_values) * mask, 2) ** 2, decoders_values

    def forward_test(self, inputs_values, mask) -> torch.Tensor:
        """input_values.shape = (batch_size, input_dim)"""
        # P(X | z) = N( A mlp(z) + a, B mlp(z) + b)
        # P(z) = N(0, I)
        # Q(z | X) = Q1(z | X) * Q2(z | X) = N( C mlp(X) + c, PHI_1) * N( E mlp(X) + e, PHI_2)

        # z is modeled as a product of two independent Gaussian distributions
        z1_mean, z1_logvar = self.views_conditionalGaussianEncoders_list[0](inputs_values, mask)
        z1_var = torch.exp(z1_logvar)
        z2_mean, z2_logvar = self.views_conditionalGaussianEncoders_list[1](inputs_values, mask)
        z2_var = torch.exp(z2_logvar)

        h_mean = (z1_mean / z1_var + z2_mean / z2_var) / (1 / z1_var + 1 / z2_var)
        g_var = 1 / (1 / z1_var + 1 / z2_var)

        # Compute Kulback-Leibler : log( Q(z1 | X) Q(z2 | z1, X)  /  P(z) )
        kl_qp = self.kullback_leibler_diaggaussian_standardgaussian(h_mean, g_var)

        # if self.z1z2_orthogonal:
        #     z2_mean = z2_mean - torch.sum(z2_mean * z1_mean, dim=-1, keepdim=True) * z1_mean / torch.sum(z1_mean ** 2 + 1e-6, dim=-1, keepdim=True)

        # Cross entropy loss data
        h_mean = h_mean.repeat(self.num_of_z_sampled, 1, 1)  # .repeat(self.num_of_z_sampled,1) for MLP
        g_var = g_var.repeat(self.num_of_z_sampled, 1, 1)

        h_sample = h_mean + torch.sqrt(g_var) * torch.randn_like(h_mean)

        # compute decoders parameters
        decoders_output = self.views_conditionalGaussianDecoders(h_sample)

        decoders_values, cov_scale_tril = decoders_output

        ce_qp = self.gaussian_cross_entropy(
            decoders_values, inputs_values.repeat(self.num_of_z_sampled, 1, 1), cov_scale_tril
        )
        # ce_qp = self.gaussian_cross_entropy_manual(decoders_values, inputs_values.repeat(self.num_of_z_sampled,1, 1), cov_scale_tril)

        # regularizer
        z1z2_orthogonal_reg_term = self.z1z2_orthogonal_reg * torch.mean(
            (z1_mean.transpose(0, 2) @ z2_mean.transpose(0, 2).transpose(1, 2)) ** 2
        )
        extended_mask = mask.repeat(self.num_of_z_sampled, 1)
        return (
            torch.sum(kl_qp * mask) / torch.sum(mask),
            -torch.sum(ce_qp * extended_mask) / torch.sum(extended_mask),
            z1z2_orthogonal_reg_term,
            z1_mean,
            z2_mean,
        )

    def forward_minhyp(self, inputs_values, mask) -> torch.Tensor:
        """input_values.shape = (batch_size, input_dim)"""
        # No hypothesis except (everything diag gaussian + P(z1) * P(2) )
        # P(X | z1, z2) = N( A mlp([z1, z2]) + a, B mlp([z1, z2]) + b) OR N( A mlp((z1 + z2)) + a, B mlp((z1 + z2)) + b)
        # P(z1, z2) = P(z1) * P(z2 | z1) = N(0, I) * N(z2_mean, z2_std)
        # Q(z1, z2 | X) = Q(z1 | X) * Q(z2 | z1,X) = N( C mlp(X) + c, PHI_1) * N( E mlp(X) + e, PHI_2)

        z1_mean, z1_logvar = self.views_conditionalGaussianEncoders_list[0](inputs_values, mask)
        z1_var = torch.exp(z1_logvar)
        z1_mean = z1_mean.repeat(self.num_of_z_sampled, 1, 1)
        z1_var = z1_var.repeat(self.num_of_z_sampled, 1, 1)
        z1_sample = z1_mean + torch.sqrt(z1_var) * torch.randn_like(z1_mean)

        mask = mask.repeat(self.num_of_z_sampled, 1) if mask is not None else mask
        inputs_values = inputs_values.repeat(self.num_of_z_sampled, 1, 1)
        z2_mean, z2_logvar = self.views_conditionalGaussianEncoders_list[1](inputs_values, z1_sample, mask)
        z2_var = torch.exp(z2_logvar)
        z2_sample = z2_mean + torch.sqrt(z2_var) * torch.randn_like(z2_mean)

        # Compute Kulback-Leibler : Q log(  Q(z1,z2 | X)  / P(z1, z2) )  =  Q log(Q(z1| X) / P(z1) )  + Q log( Q(z2|z1, X) /P(z2)  )
        kl_qp_1 = self.kullback_leibler_diaggaussian_standardgaussian(z1_mean, z1_var)  #  Q log( P(z1) / Q(z1| X) )
        kl_qp_2 = self.kullback_leibler_diaggaussian_diaggaussian(
            z1_mean, z1_var, self.z2_latent_mean, self.z2_latent_var
        )  #  Q log( P(z2) / Q(z2 | z1, X) )

        kl_qp = kl_qp_1 + kl_qp_2 / self.num_of_z_sampled

        # compute decoders parameters
        if self.latent_fusion == "concat":
            z_concat = torch.cat((z1_sample, z2_sample), dim=-1)
            decoders_output = self.views_conditionalGaussianDecoders(z_concat)
        elif self.latent_fusion == "average":
            z_sample = (z1_sample + z2_sample) / 2
            decoders_output = self.views_conditionalGaussianDecoders(z_sample)
        else:
            raise ValueError("Invalid latent_fusion")
        decoders_values, cov_scale_tril = decoders_output

        ce_qp = self.gaussian_cross_entropy(decoders_values, inputs_values, cov_scale_tril)

        # regularizer
        z1z2_orthogonal_reg_term = self.z1z2_orthogonal_reg * torch.mean(
            (z1_mean.transpose(0, 2) @ z2_mean.transpose(0, 2).transpose(1, 2)) ** 2
        )
        return (
            torch.sum(kl_qp * mask) / torch.sum(mask),
            -torch.sum(ce_qp * mask) / torch.sum(mask),
            z1z2_orthogonal_reg_term,
            z1_mean,
            z2_mean,
        )

    def kullback_leibler_diaggaussian_standardgaussian(self, mean, var):
        kl_qp = (
            torch.sum(mean**2, dim=-1) + torch.sum(var, dim=-1) - self.latent_dim - torch.sum(torch.log(var), dim=-1)
        )
        kl_qp = 0.5 * kl_qp
        return kl_qp

    def kullback_leibler_diaggaussian_diaggaussian(self, mean1, var1, mean2, var2):
        var2 = var2.to(mean1.device)
        mean2 = mean2.to(mean1.device)
        kl_qp = (
            torch.sum((mean1 - mean2) ** 2 / var2, dim=-1)
            + torch.sum(var1 / var2, dim=-1)
            - self.latent_dim
            - torch.sum(torch.log(var1 / var2), dim=-1)
        )
        kl_qp = 0.5 * kl_qp
        return kl_qp

    def sigmoid_cross_entropy(self, pred, truth, mask):
        """labels : ground truth i.e data to reconstruct
        truth :   prediction i.e data reconstructed
        """
        return torch.mean(torch.sum(mask * (truth * torch.log(pred) + (1 - truth) * torch.log(1 - pred)), dim=1))

    def gaussian_cross_entropy(self, reconstructed, mean, cov_scale_tril):
        """
        reconstructed :  posterior mean (= prediction i.e data reconstructed)
        mean : ground truth i.e data to reconstruct
        covar_matrix : posterior covariance matrix

        Note that because of symmetry, truth and mean are interchangeable.
        """
        cov_scale_tril = cov_scale_tril + torch.eye(cov_scale_tril.shape[0], device=cov_scale_tril.device) * 1e-5
        normal_distrib = MultivariateNormal(
            loc=mean, scale_tril=cov_scale_tril, validate_args=False
        )  # validate_args=False
        return normal_distrib.log_prob(reconstructed)

        # normal_distrib = [  MultivariateNormal(loc=mean[s], covariance_matrix=(covar_matrix[s] + 1e-6) * torch.eye(self.views_dimensions[s])[None,:]) for s in range(self.num_of_views) ]
        # return torch.sum(torch.stack([torch.mean(normal_distrib[s].log_prob(truth[s])) for s in range(self.num_of_views)])) # cross entropy loss data

    def gaussian_cross_entropy_manual(self, reconstructed, mean, cov_scale_tril):

        # covariance_matrix = cov_scale_tril @ cov_scale_tril.T
        # regularized_covar = covariance_matrix + 1e-6 * torch.eye(covariance_matrix.shape[0], device=covariance_matrix.device)
        regularized_cov_scale_tril = (
            cov_scale_tril + torch.eye(cov_scale_tril.shape[0], device=cov_scale_tril.device) * 1e-5
        )

        # Compute inverse and log determinant of the covariance matrix
        scale_tril_inv = torch.linalg.inv(regularized_cov_scale_tril)
        covar_inv = scale_tril_inv @ scale_tril_inv.T
        log_det_covar = 2 * torch.logdet(regularized_cov_scale_tril)

        # Compute the difference
        delta = reconstructed - mean

        # Quadratic term
        quad_term = ((delta @ covar_inv) * delta).sum(axis=-1)

        # Dimensionality of the distribution
        dim = cov_scale_tril.shape[1]

        # Log probability
        log_prob = -0.5 * (
            quad_term + log_det_covar + dim * torch.log(torch.tensor(2 * torch.pi, device=cov_scale_tril.device))
        )  # -0.5 * ( [  (x - \mu)^T COVAR^-1  (x - \mu)] + [log |COVAR|] + [dim*log(2*pi)]  )
        return log_prob


class ConditionalGaussianEncodersMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=3,
        hidden_layer_size=1024,
        num_of_layers=3,
        dropout=0.1,
        use_common_hidden_layer=True,
        l2_reg=0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_of_layers = num_of_layers
        self.dropout = dropout
        self.use_common_hidden_layer = use_common_hidden_layer
        self.l2_reg = l2_reg

        if self.use_common_hidden_layer:
            self.layers, self.layers_dropout = self._create_layers_and_dropout()
        else:
            self.layers_mean, self.layers_mean_dropout = self._create_layers_and_dropout()
            self.layers_logvar, self.layers_logvar_dropout = self._create_layers_and_dropout()

        self.mean = nn.Linear(self.hidden_layer_size, self.output_dim)
        self.logvar = nn.Linear(self.hidden_layer_size, self.output_dim)

        # Initialize mean and logvar with zeros
        init.zeros_(self.mean.weight)
        init.zeros_(self.mean.bias)
        init.zeros_(self.logvar.weight)
        init.zeros_(self.logvar.bias)

    def _create_layers_and_dropout(self):
        layers = nn.ModuleList(
            [
                nn.Linear(self.input_dim if i == 0 else self.hidden_layer_size, self.hidden_layer_size)
                for i in range(self.num_of_layers)
            ]
        )
        layers_dropout = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_of_layers)])
        return layers, layers_dropout

    def forward(self, view, variable_for_compatibility=None):
        if self.use_common_hidden_layer:
            output = view
            for layer, dropout in zip(self.layers, self.layers_dropout):
                output = dropout(torch.relu(layer(output)))
            return self.mean(output), self.logvar(output)
        else:
            output_mean = view
            output_logvar = view
            for layer_mean, layer_logvar, dropout_mean, dropout_logvar in zip(
                self.layers_mean, self.layers_logvar, self.layers_mean_dropout, self.layers_logvar_dropout
            ):
                output_mean = dropout_mean(torch.relu(layer_mean(output_mean)))
                output_logvar = dropout_logvar(torch.relu(layer_logvar(output_logvar)))
            return self.mean(output_mean), self.logvar(output_logvar)


class ConditionalGaussianDecodersMLP(nn.Module):
    def __init__(
        self,
        input_dim=3,
        output_dim=3,
        hidden_layer_size=1024,
        num_of_layers=3,
        covariance_matrix_type="diagonal",
        dropout=0.1,
        l2_reg=0,
        scalar_decoder_std=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_of_layers = num_of_layers
        self.covariance_matrix_type = covariance_matrix_type
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.scalar_decoder_std = scalar_decoder_std

        self.layers = self._create_layers()
        self.layers_dropout = self._create_dropout_layers()
        self.output_layer = nn.Linear(self.hidden_layer_size, self.output_dim, bias=True)
        self.scale_tril = self._create_scale_tril()

    def _create_layers(self):
        return nn.ModuleList(
            [
                nn.Linear(self.input_dim if i == 0 else self.hidden_layer_size, self.hidden_layer_size)
                for i in range(self.num_of_layers)
            ]
        )

    def _create_dropout_layers(self):
        return nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_of_layers)])

    def _create_scale_tril(self):
        if self.covariance_matrix_type == "scalar":
            return nn.Parameter(torch.ones(1, 1))
        elif self.covariance_matrix_type == "diagonal":
            return nn.Parameter(torch.ones(self.output_dim))
        elif self.covariance_matrix_type == "full":
            tampon = torch.eye(self.output_dim)
            lower_triangular_indices = torch.tril_indices(self.output_dim, self.output_dim, offset=0)
            return nn.Parameter(
                tampon[lower_triangular_indices[0], lower_triangular_indices[1]]
            )  # init to identity matrix
        else:
            raise ValueError("Invalid covariance matrix type")

    def forward(self, view, mask_not_used=None):
        output = view
        if self.num_of_layers > 0:
            for layer, dropout in zip(self.layers, self.layers_dropout):
                output = dropout(torch.relu(layer(output)))

        mean_output = self.output_layer(output)
        # covariance_matrix = self._build_covariance_matrix()
        cov_scale_tril = self._build_scale_tril()
        return mean_output, cov_scale_tril  # covariance_matrix

    def _build_covariance_matrix(self):
        if self.covariance_matrix_type == "diagonal":
            return torch.diag(self.scale_tril**2)
        elif self.covariance_matrix_type == "scalar":
            return (self.scale_tril**2) * torch.eye(self.output_dim)
        elif self.covariance_matrix_type == "full":
            lower_triangular_indices = torch.tril_indices(self.output_dim, self.output_dim, offset=0)
            cov_scale_tril = torch.zeros(self.output_dim, self.output_dim)
            cov_scale_tril[lower_triangular_indices[0], lower_triangular_indices[1]] = self.scale_tril
            return cov_scale_tril @ cov_scale_tril.T

    def _build_scale_tril(self):
        if self.covariance_matrix_type == "diagonal":
            return torch.diag(self.scale_tril)
        elif self.covariance_matrix_type == "scalar":
            return (self.scale_tril**2) * torch.eye(self.output_dim)
        elif self.covariance_matrix_type == "full":
            lower_triangular_indices = torch.tril_indices(self.output_dim, self.output_dim, offset=0)
            cov_scale_tril = torch.zeros(self.output_dim, self.output_dim)
            cov_scale_tril[lower_triangular_indices[0], lower_triangular_indices[1]] = self.scale_tril
            return cov_scale_tril


class ConditionalGaussianEncodersAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        z1_dim,
        output_dim=3,
        hidden_layer_size=1024,
        num_of_layers=3,
        dropout=0.1,
        use_common_hidden_layer=True,
        l2_reg=0,
    ):
        super().__init__()

        self.input_dim = input_dim  # Dimension of X
        self.z1_dim = z1_dim  # Dimension of z1
        self.output_dim = output_dim  # Dimension of z2
        self.hidden_layer_size = hidden_layer_size
        self.num_of_layers = num_of_layers
        self.dropout = dropout
        self.use_common_hidden_layer = use_common_hidden_layer
        self.l2_reg = l2_reg

        # Attention mechanism (z1 attends to X)
        self.query_proj = nn.Linear(z1_dim, hidden_layer_size)  # Transform z1 to query
        self.key_proj = nn.Linear(input_dim, hidden_layer_size)  # Transform X to key
        self.value_proj = nn.Linear(input_dim, hidden_layer_size)  # Transform X to value
        self.attn_out_proj = nn.Linear(hidden_layer_size, hidden_layer_size)  # Output projection

        # MLP layers for Gaussian parameters
        if self.use_common_hidden_layer:
            self.layers, self.layers_dropout = self._create_layers_and_dropout()
        else:
            self.layers_mean, self.layers_mean_dropout = self._create_layers_and_dropout()
            self.layers_logvar, self.layers_logvar_dropout = self._create_layers_and_dropout()

        self.mean = nn.Linear(self.hidden_layer_size, self.output_dim)
        self.logvar = nn.Linear(self.hidden_layer_size, self.output_dim)

        # Initialize mean and logvar with zeros
        init.zeros_(self.mean.weight)
        init.zeros_(self.mean.bias)
        init.zeros_(self.logvar.weight)
        init.zeros_(self.logvar.bias)

    def _create_layers_and_dropout(self):
        layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_layer_size if i == 0 else self.hidden_layer_size, self.hidden_layer_size)
                for i in range(self.num_of_layers)
            ]
        )
        layers_dropout = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_of_layers)])
        return layers, layers_dropout

    def forward(self, x, z1, mask=None):
        """
        x: (batch_size, input_dim) -> Feature vector X
        z1: (batch_size, z1_dim) -> Latent variable z1
        """
        # Compute attention weights
        query = self.query_proj(z1)  # (batch_size, seq_length, hidden_dim)
        key = self.key_proj(x).transpose(1, 2)  #  (batch_size, hidden_dim, seq_length)
        value = self.value_proj(x)  # (batch_size, seq_length, hidden_dim)

        # Compute raw attention scores
        attn_logits = torch.bmm(query, key) / (self.hidden_layer_size**0.5)  # (batch_size, seq_length, seq_length)

        # Apply mask (assuming mask shape is (batch_size, seq_length))
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_length) to broadcast over queries
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (batch_size, seq_length, seq_length)

        # Compute attended values
        attended_x = torch.bmm(attn_weights, value)  # (batch_size, seq_length, hidden_dim)

        # Apply a projection to the attention output
        attended_x = self.attn_out_proj(attended_x)

        if self.use_common_hidden_layer:
            output = attended_x
            for layer, dropout in zip(self.layers, self.layers_dropout):
                output = dropout(torch.relu(layer(output)))
            return self.mean(output), self.logvar(output)
        else:
            output_mean = attended_x
            output_logvar = attended_x
            for layer_mean, layer_logvar, dropout_mean, dropout_logvar in zip(
                self.layers_mean, self.layers_logvar, self.layers_mean_dropout, self.layers_logvar_dropout
            ):
                output_mean = dropout_mean(torch.relu(layer_mean(output_mean)))
                output_logvar = dropout_logvar(torch.relu(layer_logvar(output_logvar)))
            return self.mean(output_mean), self.logvar(output_logvar)


# main to test
if __name__ == "__main__":
    config = FVAEConfig()
    # ---------------------------------------------
    # Data creation

    # Création de données factices
    batch_size = 10
    audio_dim = 10
    input_hidden_size = 14
    inputs_values = torch.randn(batch_size, audio_dim, input_hidden_size)

    mask = torch.zeros(batch_size, audio_dim, dtype=torch.int)  # Initialize mask with 0s
    for i in range(batch_size):
        split_point = torch.randint(low=1, high=audio_dim, size=(1,)).item()  # Random split point
        mask[i, :split_point] = 1

    # Configuration
    config = FVAEConfig(
        input_dim=input_hidden_size,
        #
        latent_dim=7,
        num_of_z_sampled=4,
        latent_fusion="concat",
        #
        encoder_nn_type="mlp",
        enc_num_of_layer=3,
        enc_hidden_layer=1024,  # scalar or list for each layer
        enc_dropout=0.1,
        encoder_use_common_hidden_layer=True,  # for mean and logvar
        enc_l2_reg=0,
        #
        decoder_nn_type="mlp",
        dec_num_of_layer=3,
        dec_hidden_layer=1024,  # scalar or list for each layer
        dec_dropout=0.5,
        dec_l2_reg=0,
        scalar_decoder_std=True,
        # Position of z2 in latent space
        z2_latent_mean=1.5,
        z2_latent_std=1,
        # Regularizer
        z1z2_orthogonal_reg=0,
    )

    # model
    model = FVAE(config)
    # Passe en avant
    kl_qp, minus_ce_qp, z1z2_orthogonal_reg_term, z1_mean, z2_mean = model(inputs_values, mask)
    minus_elbo = kl_qp + minus_ce_qp
    print("kl_qp:", kl_qp)
    print("- ce_qp:", minus_ce_qp)
    print("minus elbo (minimize this exactly):", minus_elbo)
    ################ same test with self.latent_fusion = "average" ###############
    config.latent_fusion = "average"
    model = FVAE(config)
    kl_qp, minus_ce_qp, z1z2_orthogonal_reg_term, z1_mean, z2_mean = model(inputs_values, mask)
    minus_elbo = kl_qp + minus_ce_qp
    print("kl_qp:", kl_qp)
    print("- ce_qp:", minus_ce_qp)
    print("minus elbo (minimize this exactly):", minus_elbo)

    ################ same test with self.force_test_vae = True ###############
    config.latent_fusion = "concat"
    config.force_test_vae = True
    model = FVAE(config)
    kl_qp, minus_ce_qp, z1z2_orthogonal_reg_term, z1_mean, z2_mean = model(inputs_values, mask)
    minus_elbo = kl_qp + minus_ce_qp
    print("kl_qp:", kl_qp)
    print("- ce_qp:", minus_ce_qp)
    print("minus elbo (minimize this exactly):", minus_elbo)

    ################ same test with self.force_minhyp ###############
    config.force_test_vae = False
    config.force_minhyp = True
    model = FVAE(config)
    kl_qp, minus_ce_qp, z1z2_orthogonal_reg_term, z1_mean, z2_mean = model(inputs_values, mask)
    minus_elbo = kl_qp + minus_ce_qp
    print("kl_qp:", kl_qp)
    print("- ce_qp:", minus_ce_qp)
    print("minus elbo (minimize this exactly):", minus_elbo)
