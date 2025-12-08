# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# generated using Gemini Fast on 5 Dec 2025
# adapted from examples/speechlm2/salm/train_salm.py to run on a single GPU
# without DDP

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from math import ceil


def get_batches_per_epoch(cfg):
    grad_acc_steps = cfg['trainer']['accumulate_grad_batches']
    train_ds_cfg = cfg['data']['train_ds']
    train_manifest_file = train_ds_cfg['input_cfg'][0]['manifest_filepath']
    batch_size = train_ds_cfg['batch_size']
    with open(train_manifest_file, 'r') as f:
        num_records = sum(1 for _ in f)
    batches_per_epoch = ceil(num_records/batch_size)
    return batches_per_epoch

@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("medium")

    batches_per_epoch = get_batches_per_epoch(cfg)
    max_epochs = cfg['trainer']['max_epochs']
    cfg['trainer']['val_check_interval'] = batches_per_epoch
    cfg['model']['scheduler']['T_max'] = batches_per_epoch * max_epochs
    
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = SALM(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(
        cfg.data,
        tokenizer=model.tokenizer,
        dataset=dataset,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()