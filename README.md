<!--
SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch

SPDX-License-Identifier: MIT
-->

# 🧬 FVAE-LoRA: Latent Space Factorization in LoRA

[![Paper](https://img.shields.io/badge/arXiv-2510.19640-B31B1B.svg)](https://arxiv.org/abs/2510.19640)
[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **FVAE-LoRA**, introduced in our NeurIPS 2025 paper: **"Latent Space Factorization in LoRA"**.

FVAE-LoRA uses a Variational Autoencoder (VAE) to split the LoRA latent space into two:
1. 🎯 **Task-salient features**: Dedicated to your specific downstream task.
2. 🌪️ **Residual information**: Captures the remaining variance.

**The result?** Better performance across text, audio, and image tasks compared to standard LoRA. 🚀

---

## 📑 Contents
- [Overview](#-overview)
- [Quick Start](#-quick-start-highlights)
- [Installation](#-installation)
- [Image Classification Experiments](#️-image-classification-experiments)
- [Repository Structure](#-repository-structure)
- [PEFT Library Modifications](#peft-library-modifications)
- [Citation](#-citation)
- [Contact](#-contact)
- [License](#license)

---

## 🔍 Overview

FVAE-LoRA is a Parameter-Efficient Fine-Tuning (PEFT) method that enhances LoRA's expressiveness through latent space factorization. This repository includes:
- 🛠️ **Modified 🤗 PEFT Library**: An extended version of Hugging Face PEFT with built-in FVAE-LoRA support.
- 🖼️ **Image Classification Suite**: Everything you need to reproduce our results on ViT.

---

## ✨ Quick Start (Highlights)

FVAE-LoRA is designed to be a **drop-in replacement** for standard PEFT methods. If you know how to use Hugging Face, you already know how to use FVAE-LoRA.

```python
from peft import FVAEPEFTConfig, get_peft_model
from transformers import AutoModelForImageClassification

# 1. Define your FVAE-LoRA config
fvae_peft_config = FVAEPEFTConfig(
    peft_type="FVAE_PEFT",
    latent_dim=16,  # latent dim
    latent_fusion="concat",
    enc_num_of_layer=1,
    enc_hidden_layer=16,
    enc_dropout=0.1,
    encoder_use_common_hidden_layer=True,
    dec_num_of_layer=3,
    dec_hidden_layer=128,
    z2_latent_mean=1.5,
    z2_latent_std=1,
    z1z2_orthogonal_reg=0,
    lambda_downstream=1000,
    lambda_reconstruction=1,
    lambda_z2_l2=1,
    lambda_z1_l2=1,
    lambda_kl_z1=1,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)

# 2. Load any HF and PEFT supported model
num_labels = 42
model_name_or_path = "google/vit-base-patch16-224-in21k"
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
)
model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

# 3. Convert to FVAE-LoRA 🪄
model = get_peft_model(model, fvae_peft_config)

model.print_trainable_parameters()
# Train as usual!
```

---

## ⚙️ Installation

### 1. Clone & Environment
```bash
git clone https://github.com/idiap/FVAE-LoRA.git
cd FVAE-LoRA

conda env create -f env.yaml
conda activate fvae-lora
pip install -r requirements.txt
```

### 2. Install the Modified PEFT 🛠️
You **must** install the local version of PEFT included in this repo:
```bash
pip install -e ./peft
```

### 3. Path Configuration
Update `path_constants.py` with your local directories. 
> 💡 **Tip:** This is **required** for reproducing the paper's image experiments but **optional** for custom usage described in [Quick Start](#-quick-start-highlights).

#### ViT model setup (for image experiments)

To run the image experiments:
1. Download the ViT model from: `google/vit-base-patch16-224-in21k` (Hugging Face).
2. Inside your `LARGE_MODELS_PATH` directory, create a folder named:
   `vit-base-patch16-224-in21k`
3. Place the downloaded model files inside that folder.

---

## 🖼️ Image Classification Experiments
We provide scripts to replicate image classification results on multiple benchmark datasets.

### Datasets
The following datasets are supported (automatically downloaded from Hugging Face 🤗):
- DTD - [tanganke/dtd](https://huggingface.co/datasets/tanganke/dtd)
- EuroSAT - [tanganke/eurosat](https://huggingface.co/datasets/tanganke/eurosat)
- GTSRB - [tanganke/gtsrb](https://huggingface.co/datasets/tanganke/gtsrb)
- RESISC45 - [tanganke/resisc45](https://huggingface.co/datasets/tanganke/resisc45)
- SUN397 - [tanganke/sun397](https://huggingface.co/datasets/tanganke/sun397)
- SVHN - [ufldl-stanford/svhn](https://huggingface.co/datasets/ufldl-stanford/svhn)

### 📉 Training FVAE-LoRA
Run the full suite across 3 seeds (1, 2, 42):
```bash
bash scripts/train_image_fvae_lora.sh
```

> [!IMPORTANT]
> - **SLURM:** The scripts default to SLURM. If running locally, remove the submission commands from the `*.sh` files.
> - **Project Name:** Replace `<your-project>` in the scripts with your actual project name.

### 🎛️ Hyperparameter Tuning

The FVAE-LoRA uses several loss components controlled by `lambda` hyperparameters:

- `--fvae_lambda_downstream`: Weight for the downstream task loss (default: 1000)
- `--fvae_lambda_reconstruction`: Weight for the reconstruction loss (default: 1)
- `--fvae_lambda_kl_z1`: Weight for the KL divergence on z1 (default: 1)
- `--fvae_lambda_z2_l2`: L2 regularization on z2 (default: 1)
- `--fvae_lambda_z1_l2`: L2 regularization on z1 (default: 1)

The secret sauce is in the `lambda` weights. For new tasks, we recommend starting with these sets apart from the default:
1. `(1000, 0.1, 1, 1, 1)`
2. `(1000, 0.1, 10, 1, 1)`

*Refer to **Section G** in the paper's appendix for a detailed practical guide on tuning these values.*

### Training Baseline Methods

For comparison, scripts are provided for other PEFT methods:

```bash
# Standard LoRA
# supports: pissa, rslora, dora, olora
# change fine_tuning_method="peft" # peft, pissa, rslora, dora, olora inside the bash script.
bash scripts/train_image_peft.sh

# Full fine-tuning
bash scripts/train_image_full_ft.sh
```
### 📊 Analyzing Results
Aggregate your results into a clean summary:
```bash
python prepare_results_images.py \
    --max-depth 2 \
    --exp-base exp/exp_image/fvae_peft/vit-base-patch16-224-in21k/
```

Use `--max-depth 1` for experiments apart from FVAE-LoRA.

---

## 📂 Repository Structure

```text
.
├── peft/                         # 🛠️ Modified PEFT library (core logic)
├── scripts/                      # 📜 Bash scripts for training & baselines
│   ├── train_image_fvae_lora.sh  # FVAE-LoRA training
│   ├── train_image_peft.sh       # LoRA and variants training
│   └── train_image_full_ft.sh    # Full fine-tuning baseline
├── image_main.py                 # 🚀 Main entry point for image experiments
├── image_model.py                # 🧩 Model wrapper with PEFT integration
├── image_datamodule.py           # 📊 PyTorch Lightning data module
├── prepare_results_images.py     # 📈 Results analysis script
├── path_constants.py             # ⚙️ Path configuration
├── requirements.txt              # Python dependencies
├── env.yaml                      # Conda environment specification
└── README.md                     # README
```

---

## PEFT Library Modifications

The included PEFT library is based on Hugging Face's PEFT with the following additions:

- `FVAEPEFTConfig`: Configuration class for FVAE-LoRA parameters
- FVAE-LoRA implementation with factorized latent space
- Support for variational inference in the LoRA framework

See [peft/](peft/) for the complete modified library.

---

## 📝 Citation

If you use this code or find our work helpful, please cite us:

```bibtex
@misc{kumar2025latentspacefactorizationlora,
      title={Latent Space Factorization in LoRA}, 
      author={Shashi Kumar and Yacouba Kaloga and John Mitros and Petr Motlicek and Ina Kodrasi},
      year={2025},
      eprint={2510.19640},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.19640}, 
}
```

---

## 🤝 Contact

📧 **Questions?** Open an issue or reach out at
- Shashi Kumar (shashi.kumar@idiap.ch)
- Yacouba Kaloga (yacouba.kaloga@idiap.ch) 

---

## ⚖️ License

This project is released under the MIT License. See the [LICENSES/MIT.txt](LICENSES/MIT.txt) file for details.

The modified PEFT library retains its original Apache 2.0 License - see [peft/LICENSE](peft/LICENSE).

For third-party dependencies retain their respective licenses.

---

*Built with ❤️ at the Idiap Research Institute.*