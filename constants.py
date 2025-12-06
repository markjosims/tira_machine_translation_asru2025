import os

DATASET_DIR = os.environ.get("DATASETS", "data/")
MODEL_DIR = os.environ.get("MODELS", "models/")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_CHECKPOINT = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = os.path.join(MODEL_DIR, "tira_mbart_finetuned")
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, "tira_mt_tokenized_train")
VAL_DATA_PATH = os.path.join(DATASET_DIR, "tira_mt_tokenized_validation")

ALLOPHANT_TRAIN_PATH = os.path.join(DATASET_DIR, "tira_allophant_mt_train")
ALLOPHANT_VAL_PATH = os.path.join(DATASET_DIR, "tira_allophant_mt_validation")

ASR_DS_PATH = os.path.join(DATASET_DIR, "tira_asr_translated")
NEMO_AUDIO_DIR = os.path.join(DATASET_DIR, "nemo")

CSV_FILE = "tira_sentences.csv"
MODEL_CHECKPOINT = "facebook/mbart-large-50-many-to-many-mmt"

SRC_LANG_CODE = "sw_KE"
TGT_LANG_CODE = "en_XX"
