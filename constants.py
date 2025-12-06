import os

DATASET_DIR = os.environ.get("DATASETS")
MODEL_DIR = os.environ.get("MODELS")

MODEL_CHECKPOINT = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = os.path.join(MODEL_DIR, "tira_mbart_finetuned")
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, "tira_mt_tokenized_train")
VAL_DATA_PATH = os.path.join(DATASET_DIR, "tira_mt_tokenized_validation")

CSV_FILE = "tira_sentences.csv"
MODEL_CHECKPOINT = "facebook/mbart-large-50-many-to-many-mmt"

SRC_LANG_CODE = "sw_KE"
TGT_LANG_CODE = "en_XX"
