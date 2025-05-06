import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from birdie_rl import Birdie
from birdie_rl.example_usage.ul2_config import ul2_config


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2Small(nn.Module):
    def __init__(self, vocab_size=50257, ctx_len=1024, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, ctx_len, embed_dim))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, label_ids=None, segment_ids=None):
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)  # logits


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


train_file_path = "data/genres_original/jazz/train/jazz.txt"
validation_file_path = "data/genres_original/jazz/validation/jazz.txt"
dataset = load_dataset("text", data_files={"train": train_file_path, "validation": validation_file_path})

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_ds = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
val_ds = dataset["validation"].map(tokenize, batched=True, remove_columns=["text"])


def data_generator(split, worker_id, num_workers, rng_seed=0):
    np.random.seed(rng_seed)
    data = train_ds if split == "train" else val_ds
    data = data.shard(num_shards=num_workers, index=worker_id)
    data = data.shuffle(seed=rng_seed)
    for item in data:
        yield {
            "input_ids": torch.tensor(item["input_ids"]),
            "label_ids": torch.tensor(item["input_ids"]),  # causal LM loss
            "attention_mask": torch.tensor(item["attention_mask"]),
            "segment_ids": torch.zeros_like(torch.tensor(item["input_ids"]))  # dummy
        }


def reward_fn(batch, logits):
    logits = logits.view(-1, logits.size(-1))  # [B*T, V]
    labels = batch["label_ids"].view(-1)       # [B*T]
    loss = nn.functional.cross_entropy(logits, labels, reduction='none')
    return -loss  # Birdie maximizes reward


model = GPT2Small(vocab_size=tokenizer.vocab_size)

config = {
    "reward_fn": reward_fn,
    "ds": data_generator,
    "objectives": ul2_config,
    "tokenizer": tokenizer,
    "batch_size": 4,
    "sequence_length": 512,
    "num_workers": 1,
    "steps_between_evaluations": 50,
    "num_steps": 200,
    "model": model,
    "eval_split": "validation"
}

# =============== Start Training ==================
trainer = Birdie(config=config)
trainer.train()

# Save model state
save_path = "trained_models/gpt2tiny_birdie.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")