import os
import torchaudio
from tqdm import tqdm
import torch
import numpy as np
import tiktoken
import soundfile as sf
import pdb
import shutil
import random

torchaudio.set_audio_backend("soundfile")
tokenizer = tiktoken.get_encoding("o200k_base")

SOURCE_DIR = "data/genres_original/jazz"
TRAIN_DIR = "data/genres_original/jazz/train"
VAL_DIR = "data/genres_original/jazz/validation"


SAMPLE_RATE = 16000
SEQ_LENGTH = 1024

def audio_to_tokens(filepath):
	try:
		waveform, sr = torchaudio.load(filepath)
	except Exception as e:
		print(f"[SKIPPED] Failed to load {filepath}: {e}")
		return None

	if sr != SAMPLE_RATE:
		waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

	audio = waveform.mean(dim=0).numpy()
	audio = np.clip(audio, -1, 1)
	audio_str = " ".join([f"{x:.4f}" for x in audio])
	return audio_str

def process_and_save(input_dir, output_dir):
	audio_strings = ""
	for fname in tqdm(os.listdir(input_dir)):
		if not fname.lower().endswith((".wav", ".mp3")):
			continue
		in_path = os.path.join(input_dir, fname)
		audio_str = audio_to_tokens(in_path)
		if audio_str is None:
			continue
		audio_strings += audio_str + "\n"
	
	out_path = os.path.join(output_dir, "jazz.txt")
	with open(out_path, "w") as f:
		f.write(audio_strings)
		
def split_train_test():
	SPLIT_RATIO = 0.7  # 70% train, 30% validation

	# === Create directories if they don't exist ===
	os.makedirs(TRAIN_DIR, exist_ok=True)
	os.makedirs(VAL_DIR, exist_ok=True)

	# === List all audio files ===
	audio_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".wav", ".mp3", ".flac"))]

	# === Shuffle and split ===
	random.shuffle(audio_files)
	split_index = int(len(audio_files) * SPLIT_RATIO)
	train_files = audio_files[:split_index]
	val_files = audio_files[split_index:]

	# === Move files ===
	for f in train_files:
	    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TRAIN_DIR, f))

	for f in val_files:
	    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(VAL_DIR, f))

if __name__ == "__main__":
	split_train_test()
	process_and_save(input_dir=TRAIN_DIR, output_dir=TRAIN_DIR)
	process_and_save(input_dir=VAL_DIR, output_dir=VAL_DIR)