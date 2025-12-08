"""
Convert HuggingFace ASR dataset into NeMo manifest format.

Generated with Gemini Fast on 5 Dec 2025
"""

import os
import json
from datasets import load_from_disk
import soundfile as sf
from tqdm import tqdm
from constants import (
    ASR_DS_PATH, NEMO_AUDIO_DIR,
)
import uuid

# --- CONFIGURATION ---
# 3. Define the column names in your Hugging Face dataset
# This script assumes 'audio' is a dict containing 'path', and 'translation' is a string.
HF_AUDIO_COLUMN = "audio"
HF_SRC_COLUMN = "transcription"
HF_TEXT_COLUMN = "translation"

# 4. Define the target keys for the NeMo manifest (must be absolute paths)
NEMO_TASK = "ast"
NEMO_AUDIO_KEY = "audio_filepath"
NEMO_DURATION_KEY = "duration"
NEMO_SRC_KEY = "source_text"
NEMO_SRC_LANG = "sw"
NEMO_TGT_LANG = "en"
NEMO_TEXT_KEY = "text"
NEMO_PNC = 'yes'


# --- END CONFIGURATION ---

# Ensure the output directory exists
os.makedirs(NEMO_AUDIO_DIR, exist_ok=True)


def process_and_save_audio(example, audio_output_dir):
    """
    Processes one row: saves the audio array to a WAV file, calculates duration,
    and returns the necessary NeMo fields.
    """
    audio_data = example[HF_AUDIO_COLUMN]

    # 1. Extract data and rate
    audio_array = audio_data['array']
    sampling_rate = audio_data['sampling_rate']

    # 2. Define a unique file name and path
    unique_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(audio_output_dir, unique_filename)

    # 3. Extract target text
    text = str(example[HF_TEXT_COLUMN])
    src_text = str(example[HF_SRC_COLUMN])
    
    # 4. Save the audio array to a WAV file
    try:
        # Use soundfile to write the numpy array to a WAV file
        if not os.path.exists(audio_filepath):
            sf.write(audio_filepath, audio_array, sampling_rate, format='WAV')
    except Exception as e:
        # Return empty paths/zero duration if saving fails
        print(f"Error saving audio for a sample: {e}")
        return {
            "audio_filepath": "",
            "duration": 0.0,
            NEMO_TEXT_KEY: text,
            'target_lang': NEMO_TGT_LANG,
            NEMO_SRC_KEY: src_text,
            'source_lang': NEMO_SRC_LANG,
            'task': NEMO_TASK,
            'pnc': NEMO_PNC,
        }

    # 5. Calculate duration
    duration = len(audio_array) / sampling_rate

    # Return the data needed for the NeMo manifest
    return {
        "audio_filepath": os.path.abspath(audio_filepath),  # MUST be absolute path
        "duration": duration,
        NEMO_TEXT_KEY: text,
        'target_lang': NEMO_TGT_LANG,
        NEMO_SRC_KEY: src_text,
        'source_lang': NEMO_SRC_LANG,
        'task': NEMO_TASK,
        'pnc': NEMO_PNC,
    }


def convert_hf_split_to_nemo_manifest(split_name: str):
    """
    Handles the full conversion for a single split (train, validation, or test).
    """
    hf_dataset_path = os.path.join(ASR_DS_PATH, split_name)
    audio_output_dir = os.path.join(NEMO_AUDIO_DIR, split_name)
    nemo_manifest_path = os.path.join(NEMO_AUDIO_DIR, f"nemo_{split_name}_manifest.jsonl")

    # Create the split-specific audio output directory
    os.makedirs(audio_output_dir, exist_ok=True)

    print(f"\n--- Starting Conversion for '{split_name.upper()}' Split ---")
    print(f"Loading HF dataset from: {hf_dataset_path}")

    try:
        dataset = load_from_disk(hf_dataset_path)
    except Exception as e:
        print(f"Error loading dataset split '{split_name}'. Skipping. Error: {e}")
        return

    print(f"Dataset loaded with {len(dataset)} examples. Processing audio...")

    # Define a lambda function to pass the required directory to the mapping function
    map_func = lambda x: process_and_save_audio(x, audio_output_dir)

    # Use map() to apply the saving and calculation function in parallel
    processed_dataset = dataset.map(
        map_func,
        num_proc=os.cpu_count() or 1,
        remove_columns=dataset.column_names  # Keep only the new, relevant columns
    )

    # Filter out any failed samples
    processed_dataset = processed_dataset.filter(lambda x: x["audio_filepath"] != "")

    print(f"Successfully processed and filtered {len(processed_dataset)} samples.")

    # Write the resulting dataset to the JSONL manifest file
    with open(nemo_manifest_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(processed_dataset, desc=f"Writing {split_name} manifest"):
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Manifest saved to: {os.path.abspath(nemo_manifest_path)}")
    print(f"Audio files saved to: {os.path.abspath(audio_output_dir)}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    # --- Execute the conversion for each split ---
    splits = ["train", "validation", "test"]

    for split in splits:
        convert_hf_split_to_nemo_manifest(split)