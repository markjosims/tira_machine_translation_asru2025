
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from datasets import load_from_disk
from evaluate import load
from constants import (
    MODEL_CHECKPOINT, OUTPUT_DIR, TRAIN_DATA_PATH, VAL_DATA_PATH
)
import os

def main():
    print("="*40)
    print("🚀 Tira Translation Model Training Setup")
    print("="*40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")
    # Enable Mixed Precision (fp16) only if CUDA is available
    fp16_enabled = device == "cuda"
    
    if fp16_enabled:
        print("FP16 Mixed Precision: ENABLED (Crucial for VRAM savings)")

    print("\n📂 Loading tokenized data from disk...")
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Error: Data folders not found. Run prepare_mt_data.py first!")
        return
        
    train_dataset = load_from_disk(TRAIN_DATA_PATH)
    val_dataset = load_from_disk(VAL_DATA_PATH)

    print(f"\n⬇️  Loading base mBART model: {MODEL_CHECKPOINT}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    # Load tokenizer to save with the model later
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    print("\nDefining metrics...")
    def compute_metrics(eval_preds):
        metric = load("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        return result

    print("\n⚙️  Configuring training parameters...")

    batch_size = 16
    gradient_accumulation = 1

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=fp16_enabled,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        compute_metrics=compute_metrics,
        greater_is_better=True,
        metric_for_best_model="bleu",
        # dataloader_num_workers=num_workers
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"\n🔥 STARTING FINE-TUNING 🔥")
    print(f"Batch Size: {batch_size} | Accumulation: {gradient_accumulation}")
    trainer.train()

    print(f"\n✅ Training finished! Saving final model to: /{OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
