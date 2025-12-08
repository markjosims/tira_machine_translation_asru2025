import os
# --- FIX: Use the new namespace ---
import lightning.pytorch as pl 
# ----------------------------------
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.utils.exp_manager import exp_manager
from math import ceil
import torchmetrics

def get_batches_per_epoch(cfg):
    grad_acc_steps = cfg['trainer']['accumulate_grad_batches']
    train_ds_cfg = cfg['data']['train_ds']
    train_manifest_file = train_ds_cfg['manifest_filepath']
    batch_size = train_ds_cfg['batch_size']
    with open(train_manifest_file, 'r') as f:
        num_records = sum(1 for _ in f)
    batches_per_epoch = ceil(num_records/batch_size)
    return batches_per_epoch

@hydra_runner(config_path=".", config_name="canary_1b_finetune")
def main(cfg):
    batches_per_epoch = get_batches_per_epoch(cfg)
    total_steps = batches_per_epoch * cfg['trainer']['max_epochs']
    cfg['trainer']['val_check_interval']=batches_per_epoch
    cfg['trainer']['limit_train_batches']=batches_per_epoch
    # cfg['model']['optim']['sched']['T_max']=total_steps
    # using `model.scheduler` so that we load the optimizer and scheduler
    # directly in the YAML (avoids bugs w/ NeMO optimizer setup)
    cfg['model']['scheduler']['T_max']=total_steps
    
    # This now creates a lightning.pytorch.Trainer, which matches the model's type
    trainer = pl.Trainer(**cfg.trainer)
    
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Load the model
    model = EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b-flash")
    
    # Setup data
    model.setup_training_data(train_data_config=cfg.data.train_ds)
    model.setup_validation_data(val_data_config=cfg.data.validation_ds)
    
    # Setup optimization
    # model.setup_optimization(cfg.model.optim)
    
    # Train
    trainer.fit(model)

if __name__ == '__main__':
    main()