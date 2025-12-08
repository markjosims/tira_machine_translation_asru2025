import random
import torch
import lhotse
import wandb
import jiwer
import sacrebleu
from tqdm import tqdm
from nemo.collections.speechlm2.models import SALM

# --- Configuration ---
# Hyperparams to track in WandB
CONFIG = {
    "k_shots": 3,
    "model_name": "nvidia/canary-qwen-2.5b",
    "train_manifest": "nemo_train_manifest.jsonl",
    "val_manifest": "nemo_validation_manifest.jsonl",
    "seed": 42,
    "max_new_tokens": 256,
    "limit_val_samples": None # Set to integer (e.g., 100) for quick debugging
}

def build_few_shot_prompt(target_cut, example_cuts):
    """
    Constructs the interleaved prompt and audio list.
    """
    audio_paths = []
    prompt_text_parts = []
    
    # 1. Add Context Examples (The "Few-Shots")
    for cut in example_cuts:
        # Get audio path (assuming local files)
        audio_paths.append(cut.recording.sources[0].source)
        
        # Get transcript/translation
        transcript = cut.supervisions[0].text
        
        # Canary uses <|audioplaceholder|> to mark where audio is inserted
        # We explicitly state "Transcript:" to guide the model
        prompt_text_parts.append(f"Audio: <|audioplaceholder|>\nEnglish translation: {transcript}")

    # 2. Add Target (The "Query")
    audio_paths.append(target_cut.recording.sources[0].source)
    # Leave the prompt open-ended for the model to complete
    prompt_text_parts.append(f"Audio: <|audioplaceholder|>\nTranslate the following to English:")

    # Join with double newlines for clear separation
    full_prompt_text = "\n\n".join(prompt_text_parts)
    
    return full_prompt_text, audio_paths

def main():
    # 1. Initialize WandB
    run = wandb.init(
        project="tira_mt_asru2025",
        config=CONFIG,
        name=f"k{CONFIG['k_shots']}-sicl"
    )
    
    # Set seed for reproducibility of example selection
    random.seed(CONFIG['seed'])

    # 2. Load Data
    print(f"Loading Lhotse CutSets...")
    train_cuts = lhotse.CutSet.from_file(CONFIG['train_manifest'])
    val_cuts = lhotse.CutSet.from_file(CONFIG['val_manifest'])
    
    # Optional: Debug with smaller set
    if CONFIG['limit_val_samples']:
        val_cuts = val_cuts.subset(first=CONFIG['limit_val_samples'])

    # 3. Load Model
    print(f"Loading Model: {CONFIG['model_name']}...")
    model = SALM.from_pretrained(CONFIG['model_name'])
    model.eval().cuda()

    predictions = []
    references = []
    
    # Create a WandB Table to visualize results
    # We will log the Prompt, Target Audio, Reference, and Prediction
    results_table = wandb.Table(columns=["audio_id", "prompt_text", "reference", "prediction"])

    # 4. Evaluation Loop
    print(f"Starting {CONFIG['k_shots']}-shot evaluation on {len(val_cuts)} samples...")
    
    for i, target_cut in enumerate(tqdm(val_cuts)):
        # A. Sample k random examples from training set
        # Using .sample() on large manifests might be slow; 
        # for huge data, consider pre-loading a small "support set" into memory.
        examples = train_cuts.sample(n_cuts=CONFIG['k_shots']) 
        
        # B. Construct Prompt
        prompt_text, audio_paths = build_few_shot_prompt(target_cut, examples)
        
        # C. Generate
        generation_input = [{
            "role": "user",
            "content": prompt_text,
            "audio": audio_paths 
        }]
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    prompts=[generation_input], 
                    max_new_tokens=CONFIG['max_new_tokens']
                )
            
            # D. Decode
            # Strip special tokens to clean up output
            predicted_text = model.tokenizer.ids_to_text(output_ids[0].cpu().tolist())
            target_text = target_cut.supervisions[0].text
            
            predictions.append(predicted_text)
            references.append(target_text)
            
            # Add to WandB Table
            results_table.add_data(
                target_cut.id,
                prompt_text,
                target_text,
                predicted_text
            )
            
        except Exception as e:
            print(f"Error generating for cut {target_cut.id}: {e}")
            continue

    # 5. Compute Metrics
    print("Computing metrics...")
    
    # WER (Word Error Rate)
    wer_score = jiwer.wer(references, predictions)
    
    # BLEU (Standard implementation via SacreBLEU)
    # SacreBLEU expects references as a list of lists: [[ref1, ref2], [ref1_alt, ref2_alt]]
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score

    print(f"Final WER: {wer_score:.4f}")
    print(f"Final BLEU: {bleu_score:.2f}")

    # 6. Log to WandB
    wandb.log({
        "wer": wer_score,
        "bleu": bleu_score,
        "results_table": results_table
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()