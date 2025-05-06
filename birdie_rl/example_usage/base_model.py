import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from birdie_rl import Birdie
from birdie_rl.example_usage.ul2_config import ul2_config
from datasets import load_dataset
import tiktoken
import accelerate
import numpy as np
from tqdm import tqdm

# Rotary Positional Embeddings (RoPE) implementation
def get_rotary_embeddings(dim, max_seq_len, theta=10000.0, device="cuda"):
    """
    Compute rotary positional embeddings (RoPE) for a given dimension and sequence length.
    Returns: (max_seq_len, dim) tensor of sin/cos embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float().unsqueeze(1)
    angles = positions * freqs
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return torch.stack([cos, sin], dim=-1)  # (max_seq_len, dim//2, 2)

def apply_rotary_emb(x, rotary_emb):
    """
    Apply rotary embeddings to query/key tensors.
    x: (batch, seq_len, num_heads, head_dim)
    rotary_emb: (max_seq_len, head_dim//2, 2)
    """
    batch, seq_len, num_heads, head_dim = x.shape
    x = x.view(batch, seq_len, num_heads, head_dim // 2, 2)
    cos, sin = rotary_emb[:seq_len, :, 0], rotary_emb[:seq_len, :, 1]
    x_rotated = torch.stack([
        x[..., 0] * cos - x[..., 1] * sin,
        x[..., 0] * sin + x[..., 1] * cos
    ], dim=-1).view(batch, seq_len, num_heads, head_dim)
    return x_rotated

# New Transformer model
class PrefixLMTransformer(nn.Module):
    """
    A prefix-LM Transformer model with rotary embeddings, supporting bidirectional attention
    for prefix tokens and causal attention for suffix tokens.
    """
    def __init__(self, vocab_size=200_000, hidden_size=2048, num_layers=12, num_heads=16,
                 max_seq_len=2048, dropout=0.1, intermediate_size=8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Output head
        self.output_head = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))

        # Rotary embeddings
        self.rotary_emb = get_rotary_embeddings(self.head_dim, max_seq_len, device="cuda")
        self.register_buffer("rotary_emb", self.rotary_emb, persistent=False)

    def forward(self, input_ids, label_ids=None, segment_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Default segment_ids if not provided
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids, dtype=torch.long)

        # Default attention_mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Create attention mask for prefix-LM
        # Bidirectional for prefix (attention_mask=1), causal for suffix (attention_mask=0)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
        segment_mask = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)  # Same segment
        prefix_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)  # Bidirectional for prefix
        attn_mask = (prefix_mask | causal_mask) & segment_mask  # Combine masks
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, 1, seq_len, seq_len)

        # Embedding
        x = self.embedding(input_ids)

        # Apply transformer layers with rotary embeddings
        for layer in self.layers:
            # Extract query and key for rotary embeddings
            qkv = layer.self_attn.qkv_proj(x)  # Assuming access to internal qkv projection
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = apply_rotary_emb(q, self.rotary_emb)
            k = apply_rotary_emb(k, self.rotary_emb)
            qkv = torch.cat([q.view(batch_size, seq_len, -1),
                            k.view(batch_size, seq_len, -1), v], dim=-1)
            x = layer(x, src_mask=attn_mask.squeeze(1))

        # Final norm and output
        x = self.norm(x)
        logits = self.output_head(x)

        if label_ids is None:
            return logits

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,  # Ignore padding or invalid labels
            reduction="mean"
        )
        return loss

# Data generator function using HuggingFace's TinyStories dataset
def huggingface_data_generator_fn(split, worker_id, num_workers, rng_seed=0):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    ds = ds.shuffle(seed=rng_seed)
    ds = ds.shard(num_shards=num_workers, index=worker_id)
    return ds

def data_generator(split, worker_id, num_workers, rng_seed=0):
	"""
	The data_generator function will be called by each dataloading worker.
	This currently only data parallel training, where each accelerator has its own copy of the model.

	This function should return a generator for a given
	  - split (e.g., "train", "validation", "test")
	  - shards it by worker_id and num_workers
	  - shuffles the data using rng_seed
	"""

	# Load the dataset from a local file
	# ds = load_dataset("audiofolder", data_dir="data/genres_original/jazz", split=split, drop_labels=True)
	train_file_path = "data/genres_original/jazz/train/jazz.txt"
	validation_file_path = "data/genres_original/jazz/validation/jazz.txt"

	ds = load_dataset(
	    "text",
	    data_files={"train": train_file_path, "validation": validation_file_path}, split =split
	)

	# Load the TinyStories dataset from Hugging Face (Replaced this with the local dataset)
	# ds = load_dataset("roneneldan/TinyStories", split=split)
	

	# Shard the dataset among multiple workers
	ds = ds.shard(num_shards=num_workers, index=worker_id)

	# Shuffle the dataset for randomness
	ds = ds.shuffle(rng_seed)

	# Return the prepared dataset
	return ds

# Text grabber function to extract text from dataset
def text_grabber_fn(item):
    return item["text"]

# Configuration
config = {
    "batch_size": 8,
    "sequence_length": 2048,
    "num_workers": 8,
    "steps_between_evaluations": 50,
    "num_steps": 200,
    "accelerator": accelerate.Accelerator(),
    "tokenizer": tiktoken.get_encoding("o200k_base"),
    "objectives": ul2_config,
    "ds": huggingface_data_generator_fn,
    "text_grabber_fn": text_grabber_fn,
    "start_generating_paradigm": "\n<|assistant|>\n",
}

# Initialize model, optimizer, and scheduler
model = PrefixLMTransformer(
    vocab_size=config["tokenizer"].n_vocab,  # 200_000 for o200k_base
    hidden_size=2048,
    num_layers=12,
    num_heads=16,
    max_seq_len=config["sequence_length"],
    dropout=0.1,
    intermediate_size=8192
)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=config["num_steps"])
model, optimizer, scheduler = config["accelerator"].prepare(model, optimizer, scheduler)

# Initialize Birdie
birdie = Birdie(config)

# Create output directory for saving models
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

# Training loop
progress_bar = tqdm(total=config["num_steps"], desc="Training")
for step_idx in range(config["num_steps"]):
    progress_bar.update(1)

    # Periodic evaluations
    if birdie.time_for_eval(step_idx):
        model.eval()
        for (objective_name, batch) in birdie.measure_validation_losses():
            with torch.no_grad():
                loss = model(
                    input_ids=batch["input_ids"],
                    label_ids=batch["label_ids"],
                    segment_ids=batch.get("segment_ids"),
                    attention_mask=batch.get("attention_mask")
                )
                config["accelerator"].backward(loss)  # Ensure loss is on correct device
                birdie.log_validation_loss(key=objective_name, loss=loss.item(), step_idx=step_idx)
        model.train()
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step_idx}.pt")
        config["accelerator"].save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step_idx": step_idx,
            "config": config
        }, checkpoint_path)
        print(f"Checkpoint saved at step {step_idx} to {checkpoint_path}")

    # Fetch and train on the next batch
    batch = birdie.get_next_training_sample()
    loss = model(
        input_ids=batch["input_ids"],
        label_ids=batch["label_ids"],
        segment_ids=batch.get("segment_ids"),
        attention_mask=batch.get("attention_mask")
    )
    
    optimizer.zero_grad()
    config["accelerator"].backward(loss)
    optimizer.step()
    scheduler.step()

# Save final model
final_model_path = os.path.join(output_dir, "final_model.pt")
config["accelerator"].save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "step_idx": config["num_steps"],
    "config": config
}, final_model_path)
print(f"Final model saved to {final_model_path}")

# Save RewardModel
reward_model_path = os.path.join(output_dir, "reward_model.pt")
config["accelerator"].save(birdie.reward_model.state_dict(), reward_model_path)
print(f"RewardModel saved to {reward_model_path}")

print("\n" * 3, end="")
print("All done. Closing Birdie...")

# Close Birdie
birdie.close()

# Hard exit to clean up
os._exit(0)