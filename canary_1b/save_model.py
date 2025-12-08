import os
import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel

# --- CONFIGURATION ---
# 1. Provide the path to the original configuration YAML used for training
CONFIG_YAML_PATH = "canary_1b/canary_1b_ft.yaml" 
# 2. Provide the path to the specific .ckpt file you want to save
CHECKPOINT_PATH = "/home/mjsimmons/projects/tira_machine_translation_asru2025/nemo_experiments/tira_mt_asru2025/2025-12-06_11-12-28/checkpoints/epoch=29-step=5400.ckpt"
# 3. Define the desired output path
OUTPUT_NEMO_PATH = "/home/mjsimmons/projects/tira_machine_translation_asru2025/nemo_experiments/tira_mt_asru2025/model.nemo"
# 4. Path to the base model's .nemo file (e.g., the one you downloaded or saved initially)
NEMO_MODEL_PATH = "/home/mjsimmons/.cache/huggingface/hub/models--nvidia--canary-1b-flash/blobs/3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324" 
# ---------------------

import torch
from nemo.collections.asr.models import EncDecMultiTaskModel

# --- CONFIGURATION ---


# 1. Load the model structure and base weights from the .nemo file
# This correctly initializes the architecture, configuration, and tokenizer.
print(f"1. Restoring model structure and configuration from {NEMO_MODEL_PATH}...")
model = EncDecMultiTaskModel.restore_from(NEMO_MODEL_PATH, map_location='cuda')

# 2. Load the .ckpt file (which contains the weights)
# We load the entire checkpoint dictionary, not just the weights.
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda', weights_only=False)

# 3. Extract the state_dict (weights) from the checkpoint dictionary
# PyTorch Lightning stores the weights under the 'state_dict' key in the checkpoint.
state_dict = checkpoint['state_dict']

# 4. Load the weights into the restored model
# strict=False allows the model to ignore optimizer states or keys that are present 
# in the checkpoint but not in the model (e.g., metric buffers).
print("2. Applying fine-tuned weights from .ckpt file...")
model.load_state_dict(state_dict, strict=False)

# 5. Final setup for inference
model.eval().cuda()
print("✅ Model successfully loaded and updated with fine-tuned weights!")

# 3. CRITICAL: Save the consolidated .nemo file
# This packages the model's configuration (from the YAML), the weights, and 
# any required artifacts (like the tokenizer).
print(f"Saving consolidated model to {OUTPUT_NEMO_PATH}...")
model.save_to(OUTPUT_NEMO_PATH)
print("✅ .nemo file successfully created!")

# 4. Clean up (Optional)
del model
torch.cuda.empty_cache()

# 5. Verify the new file loads using the robust method:
# final_model = EncDecMultiTaskModel.restore_from(OUTPUT_NEMO_PATH)