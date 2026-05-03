"""
UPDATED Training Script with Fixed Tokenizers
Key changes:
1. Uses new tokenizers (tokenizer_en_v2, tokenizer_fr_v2)
2. Increased model size (d_model: 32 → 256)
3. Proper handling of distinct PAD/EOS tokens
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ============================================================================
# PATHS & SETUP
# ============================================================================
DATASET_PATH = "parallel_en_fr_corpus"
EN_TOKENIZER_PATH = "tokenizer_en_v2"  # ← CHANGED from tokenizer_en
FR_TOKENIZER_PATH = "tokenizer_fr_v2"  # ← CHANGED from tokenizer_fr
CHECKPOINT_DIR = "checkpoints_v2"      # ← NEW directory to avoid conflicts
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("\nLoading datasets...")
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset   = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset  = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============================================================================
# LOAD FIXED TOKENIZERS
# ============================================================================
print("\nLoading FIXED tokenizers...")
en_tokenizer = PreTrainedTokenizerFast.from_pretrained(EN_TOKENIZER_PATH)
fr_tokenizer = PreTrainedTokenizerFast.from_pretrained(FR_TOKENIZER_PATH)

# Get special token IDs
src_pad_id = en_tokenizer.pad_token_id
tgt_pad_id = fr_tokenizer.pad_token_id
bos_token_id = fr_tokenizer.bos_token_id
eos_token_id = fr_tokenizer.eos_token_id

src_vocab_size = len(en_tokenizer)
tgt_vocab_size = len(fr_tokenizer)

# ============================================================================
# CRITICAL VERIFICATION
# ============================================================================
print("\n" + "="*70)
print("TOKENIZER VERIFICATION")
print("="*70)
print(f"Source (EN) vocab size: {src_vocab_size}")
print(f"Target (FR) vocab size: {tgt_vocab_size}")
print(f"\nSource special tokens:")
print(f"  PAD: {src_pad_id}, BOS: {en_tokenizer.bos_token_id}, EOS: {en_tokenizer.eos_token_id}")
print(f"\nTarget special tokens:")
print(f"  PAD: {tgt_pad_id}, BOS: {bos_token_id}, EOS: {eos_token_id}")

# CRITICAL CHECK
if tgt_pad_id == eos_token_id:
    print("\n❌ ERROR: PAD and EOS are still the same!")
    print("   Training will fail. Please fix tokenizers first.")
    exit(1)
else:
    print(f"\n✅ PAD ({tgt_pad_id}) != EOS ({eos_token_id}) - GOOD!")
print("="*70)

# ============================================================================
# TOKENIZATION
# ============================================================================
MAX_LEN = 64

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

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_pair, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(tokenize_pair, remove_columns=val_dataset.column_names)
test_dataset  = test_dataset.map(tokenize_pair, remove_columns=test_dataset.column_names)

train_dataset.set_format('torch', columns=['src', 'tgt'])
val_dataset.set_format('torch', columns=['src', 'tgt'])
test_dataset.set_format('torch', columns=['src', 'tgt'])

# ============================================================================
# COLLATE FUNCTION
# ============================================================================
def collate_fn(batch):
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_id)

    src_mask = (src_padded != src_pad_id).unsqueeze(1).unsqueeze(2)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
    }

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ============================================================================
# BUILD MODEL WITH INCREASED SIZE
# ============================================================================
from model import build_transformer

print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)

# OLD (broken): d_model=32, d_ff=128, h=4, N=3
# NEW (fixed): Larger model with better capacity
d_model = 512   # ← CHANGED from 32 to 256 (8x larger!)
d_ff = 2048     # ← CHANGED from 128 to 1024
h = 8           # ← CHANGED from 4 to 8
N = 4           # ← CHANGED from 3 to 4
dropout = 0.1

print(f"d_model: {d_model} (was 32)")
print(f"d_ff: {d_ff} (was 128)")
print(f"heads: {h} (was 4)")
print(f"layers: {N} (was 3)")
print(f"dropout: {dropout}")

# Rough parameter count
params_estimate = (
    src_vocab_size * d_model +  # src embedding
    tgt_vocab_size * d_model +  # tgt embedding
    N * (4 * d_model * d_model + 2 * d_model * d_ff) * 2 +  # enc+dec layers
    d_model * tgt_vocab_size  # output projection
)
print(f"\nEstimated parameters: ~{params_estimate/1e6:.1f}M")
print("="*70)

model = build_transformer(
    src_vocab_size, 
    tgt_vocab_size,
    MAX_LEN, 
    MAX_LEN, 
    d_model,  # 256 
    N,        # 4
    h,        # 8
    dropout, 
    d_ff      # 1024
)
model.to(device)

# ============================================================================
# TRAINING SETUP
# ============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=0.1)  # Use tgt_pad_id!

# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

# ============================================================================
# BEAM SEARCH (Fixed)
# ============================================================================
def beam_search(model, src_tokens, src_mask, beam_width=4, max_len=64, min_len=3, temperature=0.6):
    model.eval()
    encoder_output = model.encode(src_tokens, src_mask)
    
    beams = [(0.0, [bos_token_id])]
    finished = []
    
    for step in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_token_id:
                finished.append((score, seq))
                continue
            
            tgt_tensor = torch.tensor([seq], device=device)
            tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
            causal_mask = torch.tril(torch.ones((len(seq), len(seq)), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_mask & causal_mask
            
            decoder_output = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)
            logits = model.project(decoder_output)
            next_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            log_probs = torch.log_softmax(next_logits / temperature, dim=-1)
            
            # Forbid EOS as first token
            if len(seq) == 1:
                log_probs[eos_token_id] = -float('inf')
            
            # Add EOS bonus after min_len
            if len(seq) >= min_len and seq[-1] != eos_token_id:
                log_probs[eos_token_id] += 0.5  # EOS bonus
            
            # Apply repetition penalty
            for token_id in set(seq[1:]):  # Exclude BOS
                log_probs[token_id] -= 0.5  # Penalty for repeating
            
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            for k in range(beam_width):
                token_id = top_indices[k].item()
                log_prob = top_log_probs[k].item()
                
                new_score = score + log_prob
                new_seq = seq + [token_id]
                new_beams.append((new_score, new_seq))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        
        if all(seq[-1] == eos_token_id for _, seq in beams):
            break
    
    for score, seq in beams:
        if seq[-1] != eos_token_id:
            finished.append((score, seq))
    
    if not finished:
        finished = beams
    
    best_seq = max(finished, key=lambda x: x[0])[1]
    if best_seq and best_seq[0] == bos_token_id:
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == eos_token_id:
        best_seq = best_seq[:-1]
    return best_seq

# ============================================================================
# DEBUGGING HELPER
# ============================================================================
def debug_sample_prediction(model, sample_en="i am a cat", sample_fr="je suis un chat"):
    """Debug if model generates EOS tokens"""
    print("\n" + "="*70)
    print("DEBUGGING SAMPLE PREDICTION")
    print("="*70)
    print(f"Input: '{sample_en}'")
    
    model.eval()
    
    src_tokens = en_tokenizer(sample_en, add_special_tokens=True, return_tensors='pt').to(device)
    src_ids = src_tokens['input_ids']
    src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2).bool()
    
    encoder_out = model.encode(src_ids, src_mask)
    
    # Generate step-by-step
    tgt_ids = [bos_token_id]
    print(f"\nStep-by-step generation:")
    print(f"  Start: [BOS={bos_token_id}]")
    
    found_eos = False
    for step in range(20):
        tgt_tensor = torch.tensor([tgt_ids]).to(device)
        tgt_len = len(tgt_ids)
        tgt_pad_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        tgt_mask = tgt_pad_mask.unsqueeze(0) & causal_mask.unsqueeze(0)
        
        with torch.no_grad():
            out = model.decode(tgt_tensor, encoder_out, tgt_mask, src_mask)
            logits = model.project(out)
            probs = torch.softmax(logits[0, -1], dim=0)
            
            # Top 3 predictions
            top3_probs, top3_ids = torch.topk(probs, 3)
            
            next_token = top3_ids[0].item()
            token_str = fr_tokenizer.decode([next_token])
            
            is_special = ""
            if next_token == eos_token_id:
                is_special = " [EOS]"
                found_eos = True
            elif next_token == tgt_pad_id:
                is_special = " [PAD]"
            
            print(f"  Step {step+1}: '{token_str}' (id={next_token}, p={top3_probs[0].item():.3f}){is_special}")
            
            tgt_ids.append(next_token)
            
            if next_token == eos_token_id:
                break
    
    if found_eos:
        print(f"\n✅ Model generated EOS at step {len(tgt_ids)-1}")
    else:
        print(f"\n❌ Model did NOT generate EOS in 20 steps!")
    
    final = fr_tokenizer.decode(tgt_ids, skip_special_tokens=True)
    print(f"Translation: '{final}'")
    print(f"Expected: '{sample_fr}'")
    print("="*70)

# ============================================================================
# BLEU COMPUTATION
# ============================================================================
def compute_bleu(model, dataloader, beam_width=4, max_samples=50, min_len=3):  # ← ADD min_len
    """Compute BLEU score"""
    model.eval()
    references = []
    hypotheses = []

    for i, batch in enumerate(tqdm(dataloader, desc="BLEU evaluation")):
        if max_samples and i >= max_samples:
            break
        src = batch['src'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt = batch['tgt']

        batch_size = src.size(0)
        for j in range(batch_size):
            src_tokens = src[j:j+1]
            src_mask_j = src_mask[j:j+1]
            pred_ids = beam_search(model, src_tokens, src_mask_j, beam_width, min_len=min_len)  # ← PASS min_len

            # Get reference
            ref_ids = tgt[j].tolist()
            if ref_ids and ref_ids[0] == bos_token_id:
                ref_ids = ref_ids[1:]
            if ref_ids and ref_ids[-1] == eos_token_id:
                ref_ids = ref_ids[:-1]

            references.append([ref_ids])
            hypotheses.append(pred_ids if pred_ids else [])

    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu

# ============================================================================
# TRAINING LOOP
# ============================================================================
num_epochs = 15
best_val_loss = float('inf')
patience_counter = 0

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

for epoch in range(num_epochs):
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

        # Build target mask
        tgt_len = tgt_input.size(1)
        tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & causal_mask

        # Forward pass
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
        logits = model.project(decoder_output)

        # Loss
        loss = criterion(
            logits.contiguous().view(-1, tgt_vocab_size),
            tgt_output.contiguous().view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()
        train_progress.set_postfix({'loss': f'{loss.item():.4f}'})

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

            loss = criterion(
                logits.contiguous().view(-1, tgt_vocab_size),
                tgt_output.contiguous().view(-1)
            )
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # Print epoch summary
    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Debug prediction every few epochs
    if (epoch + 1) % 3 == 0:
        debug_sample_prediction(model)

    # Save checkpoint
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print("  -> New best model!")
    else:
        patience_counter += 1

    save_checkpoint(epoch+1, model, optimizer, avg_val_loss, is_best=is_best)

    if patience_counter >= 5:
        print(f"Early stopping after {epoch+1} epochs.")
        break

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

# Load best model
load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"), model)

# Test loss
model.eval()
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
        
        loss = criterion(
            logits.contiguous().view(-1, tgt_vocab_size),
            tgt_output.contiguous().view(-1)
        )
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# BLEU score
print("\nComputing BLEU score...")
bleu = compute_bleu(model, test_loader, beam_width=4, max_samples=100, min_len=3) 
print(f"BLEU-4 score: {bleu:.4f}")
# Final debug
debug_sample_prediction(model, "i am a student", "je suis étudiant")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"If BLEU > 0.00, your model is working!")
print(f"Expected BLEU after 15 epochs: 0.10-0.30")
print("="*70)