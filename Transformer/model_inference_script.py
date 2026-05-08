import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np

# -------------------------------
# Paths & Setup
# -------------------------------
DATASET_PATH = "parallel_en_fr_corpus"
EN_TOKENIZER_PATH = "tokenizer_en"
FR_TOKENIZER_PATH = "tokenizer_fr"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ATTENTION_VIZ_DIR = "attention_viz"
os.makedirs(ATTENTION_VIZ_DIR, exist_ok=True)

# Device: MPS for Mac, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load Datasets (directly from .arrow)
# -------------------------------
print("Loading datasets...")
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset   = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset  = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print("Train columns:", train_dataset.column_names)

# -------------------------------
# Load Tokenizers
# -------------------------------
en_tokenizer = PreTrainedTokenizerFast.from_pretrained(EN_TOKENIZER_PATH)
fr_tokenizer = PreTrainedTokenizerFast.from_pretrained(FR_TOKENIZER_PATH)

# Separate source and target pad tokens (critical)
src_pad_id = en_tokenizer.pad_token_id
tgt_pad_id = fr_tokenizer.pad_token_id

bos_token_id = fr_tokenizer.bos_token_id
eos_token_id = fr_tokenizer.eos_token_id

src_vocab_size = len(en_tokenizer)
tgt_vocab_size = len(fr_tokenizer)
# Add this after loading tokenizers
print("\n=== Tokenizer Debug Info ===")
print(f"Source pad_token_id: {src_pad_id}")
print(f"Target pad_token_id: {tgt_pad_id}")
print(f"BOS token_id: {bos_token_id}")
print(f"EOS token_id: {eos_token_id}")
print(f"Target pad_token string: '{fr_tokenizer.pad_token}'")
print(f"Target eos_token string: '{fr_tokenizer.eos_token}'")

# Check if pad and eos are the same
if tgt_pad_id == eos_token_id:
    print("❌ CRITICAL: PAD and EOS tokens are the SAME! This will break training!")

# Also check the actual tokens for the repeating word "aussi"
test_ids = fr_tokenizer("aussi", add_special_tokens=False)['input_ids']
print(f"Token IDs for 'aussi': {test_ids}")
print(f"Token string for ID {test_ids[0] if test_ids else 'N/A'}: '{fr_tokenizer.decode([test_ids[0]]) if test_ids else 'N/A'}'")
# -------------------------------
# Tokenization
# -------------------------------
MAX_LEN = 64   # increased for BPE subwords

def tokenize_pair(example):
    src = en_tokenizer(
        example['text_en'],
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding=False
    )['input_ids']
    tgt = fr_tokenizer(
        example['text_fr'],
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding=False
    )['input_ids']
    return {'src': src, 'tgt': tgt}

print("Tokenizing...")
train_dataset = train_dataset.map(tokenize_pair, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(tokenize_pair, remove_columns=val_dataset.column_names)
test_dataset  = test_dataset.map(tokenize_pair, remove_columns=test_dataset.column_names)

train_dataset.set_format('torch', columns=['src', 'tgt'])
val_dataset.set_format('torch', columns=['src', 'tgt'])
test_dataset.set_format('torch', columns=['src', 'tgt'])

# -------------------------------
# Collate Function (fixed)
# -------------------------------
def collate_fn(batch):
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_id)

    # Source mask (padding only)
    src_mask = (src_padded != src_pad_id).unsqueeze(1).unsqueeze(2)

    # Target mask will be rebuilt inside the training loop after shifting
    # For now we just return the padded target (will be shifted later)
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
    }

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# Build Transformer
# -------------------------------
from model import build_transformer   # Your model code (must store attention scores)

d_model = 32; d_ff = 128; h = 4; N = 3; dropout = 0.1
model = build_transformer(src_vocab_size, tgt_vocab_size,
                          MAX_LEN, MAX_LEN, d_model, N, h, dropout, d_ff)
model.to(device)

# -------------------------------
# Training Setup
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=0.1)  # use tgt_pad_id

# -------------------------------
# Checkpoint Helpers
# -------------------------------
def save_checkpoint(epoch, model, optimizer, val_loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"))

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

# -------------------------------
# Beam Search for Translation (fixed mask)
# -------------------------------
def beam_search(model, src_tokens, src_mask, beam_width=4, max_len=64, temperature =1.2):
    model.eval()
    encoder_output = model.encode(src_tokens, src_mask)

    beams = [(0.0, [bos_token_id])]
    finished = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_token_id:
                finished.append((score, seq))
                continue

            tgt_tensor = torch.tensor([seq], device=device)
            # Build target mask (padding + causal) using tgt_pad_id
            tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
            causal_mask = torch.tril(torch.ones((len(seq), len(seq)), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_mask & causal_mask

            decoder_output = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)
            logits = model.project(decoder_output)               # (1, seq_len, vocab)
            next_logits = logits[0, -1, :]                       # last time step
            next_logits = next_logits / temperature
            log_probs = torch.log_softmax(next_logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                new_score = score + top_log_probs[k].item()
                new_seq = seq + [top_indices[k].item()]
                new_beams.append((new_score, new_seq))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        if all(seq[-1] == eos_token_id for _, seq in beams):
            break

    for score, seq in beams:
        if seq[-1] != eos_token_id:
            finished.append((score, seq))

    best_seq = max(finished, key=lambda x: x[0])[1]
    if best_seq[0] == bos_token_id:
        best_seq = best_seq[1:]
    if best_seq[-1] == eos_token_id:
        best_seq = best_seq[:-1]
    return best_seq

def translate_sentence(model, sentence, en_tokenizer, fr_tokenizer, beam_width=4):
    tokens = en_tokenizer(sentence, add_special_tokens=True, return_tensors='pt')
    src_ids = tokens['input_ids'].to(device)
    src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2).bool()
    pred_ids = beam_search(model, src_ids, src_mask, beam_width)
    return fr_tokenizer.decode(pred_ids, skip_special_tokens=True)

# -------------------------------
# BLEU Score Computation (using token ids, not splits)
# -------------------------------
def compute_bleu(model, dataloader, beam_width=4, max_samples=None):
    model.eval()
    references = []   # list of lists of token ids
    hypotheses = []   # list of lists of token ids

    for i, batch in enumerate(tqdm(dataloader, desc="BLEU evaluation")):
        if max_samples and i >= max_samples:
            break
        src = batch['src'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt = batch['tgt']   # reference token ids on CPU

        batch_size = src.size(0)
        for j in range(batch_size):
            src_tokens = src[j:j+1]
            src_mask_j = src_mask[j:j+1]
            pred_ids = beam_search(model, src_tokens, src_mask_j, beam_width)

            # Remove BOS/EOS from reference
            ref_ids = tgt[j].tolist()
            if ref_ids and ref_ids[0] == bos_token_id:
                ref_ids = ref_ids[1:]
            if ref_ids and ref_ids[-1] == eos_token_id:
                ref_ids = ref_ids[:-1]

            references.append([ref_ids])   # corpus_bleu expects list of references per sentence
            hypotheses.append(pred_ids)

    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu

# -------------------------------
# Attention Visualization
# -------------------------------
def get_attention_weights(model, src_text, tgt_text, layer_idx=-1, head_idx=0):
    model.eval()
    src_tokens = en_tokenizer(src_text, add_special_tokens=True, return_tensors='pt').to(device)
    tgt_tokens = fr_tokenizer(tgt_text, add_special_tokens=True, return_tensors='pt').to(device)
    src_ids = src_tokens['input_ids']
    tgt_ids = tgt_tokens['input_ids']

    src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2).bool()
    tgt_mask = (tgt_ids != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
    causal = torch.tril(torch.ones((tgt_ids.size(1), tgt_ids.size(1)), device=device)).bool()
    causal = causal.unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask & causal

    # Reset stored attention scores
    for layer in model.decoder.layers:
        if hasattr(layer.cross_attention_block, 'attention_scores'):
            layer.cross_attention_block.attention_scores = None

    with torch.no_grad():
        _ = model.decode(tgt_ids, model.encode(src_ids, src_mask), tgt_mask, src_mask)

    target_layer = model.decoder.layers[layer_idx]
    attn_scores = target_layer.cross_attention_block.attention_scores
    if attn_scores is not None:
        attn_scores = attn_scores[0, head_idx].cpu().numpy()
        return attn_scores
    return None

def plot_attention(attn_matrix, src_tokens, tgt_tokens, save_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn_matrix, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='Blues', cbar=True)
    plt.xlabel('Source tokens')
    plt.ylabel('Target tokens')
    plt.title('Cross-Attention Weights')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------
# Training Loop with Validation & Checkpointing
# -------------------------------
num_epochs = 10
best_val_loss = float('inf')
patience_counter = 0
start_epoch = 0

resume = False
if resume and os.path.exists(os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")):
    start_epoch, _ = load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"), model, optimizer)
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    # ---------- TRAINING ----------
    model.train()
    total_train_loss = 0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch in train_progress:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)

        # Shift target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Build target mask for shifted input
        tgt_len = tgt_input.size(1)
        tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & causal_mask

        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
        logits = model.project(decoder_output)

        loss = criterion(logits.contiguous().view(-1, tgt_vocab_size),
                         tgt_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()
        train_progress.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)

    # ---------- VALIDATION ----------
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_len = tgt_input.size(1)
            tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
            causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_pad_mask & causal_mask

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
            logits = model.project(decoder_output)

            loss = criterion(logits.contiguous().view(-1, tgt_vocab_size),
                             tgt_output.contiguous().view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print("  -> New best model!")
    else:
        patience_counter += 1

    save_checkpoint(epoch+1, model, optimizer, avg_val_loss, is_best=is_best)
    if patience_counter >= 3:
        print(f"Early stopping after {epoch+1} epochs.")
        break

# -------------------------------
# Test Evaluation (Loss & BLEU)
# -------------------------------
load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"), model)
model.eval()

# Test loss
total_test_loss = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Loss"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_len = tgt_input.size(1)
        tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & causal_mask

        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
        logits = model.project(decoder_output)
        loss = criterion(logits.contiguous().view(-1, tgt_vocab_size),
                         tgt_output.contiguous().view(-1))
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# BLEU on test set
bleu = compute_bleu(model, test_loader, beam_width=4, max_samples=100)
print(f"BLEU-4 score (corpus, smoothed): {bleu:.4f}")

# -------------------------------
# Attention Visualization (example)
# -------------------------------
sample = next(iter(test_loader))
src_text = en_tokenizer.decode(sample['src'][0].tolist(), skip_special_tokens=True)
tgt_text = fr_tokenizer.decode(sample['tgt'][0].tolist(), skip_special_tokens=True)
print(f"\nVisualizing attention for:\nSource: {src_text}\nTarget: {tgt_text}")

attn_weights = get_attention_weights(model, src_text, tgt_text, layer_idx=-1, head_idx=0)
if attn_weights is not None:
    src_tokens = en_tokenizer.tokenize(src_text)
    tgt_tokens = fr_tokenizer.tokenize(tgt_text)
    plot_attention(attn_weights, src_tokens, tgt_tokens,
                   os.path.join(ATTENTION_VIZ_DIR, "cross_attention.png"))
    print(f"Attention map saved to {ATTENTION_VIZ_DIR}/cross_attention.png")
else:
    print("Could not retrieve attention scores. Ensure model returns cross-attention scores.")

# -------------------------------
# Live Translation Demo
# -------------------------------
def live_translate(sentence):
    return translate_sentence(model, sentence, en_tokenizer, fr_tokenizer, beam_width=4)

# Example usage (uncomment for testing):
# while True:
#     eng = input("Enter English sentence: ")
#     if eng.lower() == 'quit': break
#     print("French:", live_translate(eng))