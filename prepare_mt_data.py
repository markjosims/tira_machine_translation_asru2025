import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from constants import (
    CSV_FILE, TRAIN_DATA_PATH, VAL_DATA_PATH,
    MODEL_CHECKPOINT, SRC_LANG_CODE, TGT_LANG_CODE
)


def prepare_data():
    print("="*40)
    print("🛠️ Tira Translation Data Prep")
    print("="*40)

    print(f"\n📂 Loading raw data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        df = df[['transcription', 'translation']].dropna()
        df = df.rename(columns={'transcription': 'src_text', 'translation': 'tgt_text'})
        df = df.astype(str)
        print(f"   Found {len(df)} valid translation pairs.")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    print("\n🔄 Converting to HF Dataset format...")
    full_dataset = Dataset.from_pandas(df)
    dataset_splits = full_dataset.train_test_split(test_size=0.1, seed=42)
    datasets = DatasetDict({
        'train': dataset_splits['train'],
        'validation': dataset_splits['test']
    })

    print(f"\n⬇️  Loading tokenizer ({MODEL_CHECKPOINT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
    tokenizer.src_lang = SRC_LANG_CODE
    tokenizer.tgt_lang = TGT_LANG_CODE

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['src_text'], max_length=128, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['tgt_text'], max_length=128, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\n🔢 Tokenizing data (converting text to numbers)...")
    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets['train'].column_names)

    print(f"\n💾 Saving tokenized data to disk...")
    tokenized_datasets["train"].save_to_disk(TRAIN_DATA_PATH)
    tokenized_datasets["validation"].save_to_disk(VAL_DATA_PATH)
    print("\n✅ Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
